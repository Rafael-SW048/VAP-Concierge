from concurrent.futures import Future, ThreadPoolExecutor, wait
from concurrent.futures.process import ProcessPoolExecutor
import logging
import os
from queue import SimpleQueue
import subprocess
from threading import Event, Lock, Thread
from time import sleep
from time import perf_counter_ns as perf_counter
from typing import Dict, List

import grpc
import numpy as np
from scipy import optimize as op

from api.common.command import (
    CG_CLR,
    IF,
    IFB,
    IFB_UP,
    INSERT_KMODULE,
    TC_CLASS_REPLACE,
    TC_CLASS_ADD,
    TC_FILTER_REPLACE,
    TC_FILTER_ADD,
    TC_QDISC_ADD,
    TC_QDISC_DEL,
)
from api.common.enums import ResourceType
from api.common.typedef import Allocation, ResRequest
from api.pipeline_repr import PipelineRepr, InferDiff
import api.proto.concierge_pb2 as concierge_pb2
import api.proto.concierge_pb2_grpc as concierge_pb2_grpc

PROFILER_CPU: float = float(os.getenv("PROFILER_CPU") or 0.1)
PROFILER_MEM: float = float(os.getenv("PROFILER_MEM") or 1)
PROFILER_CG: str = os.getenv("PROFILER_CG") or "profiler"

MAX_BW: int = int(os.getenv("MAX_BW") or 60 * 1024)
MI: int = int(os.getenv("MI") or 5)
MAX_CPUSHARE: int = int(os.getenv("MAX_CPUSHARE") or 1000)
PROFILING_DELTA: int = int(os.getenv("PROFILING_DELTA") or 5000)
# BASELINE_MODE: bool = str(os.getenv("BASELINE_MODE")) == "true"
BASELINE_MODE: bool = int(os.getenv("BASELINE_MODE")) > 0
NUM_APP: int = int(os.getenv("NUM_APP") or 1)
DELTA_STEP: int = int(os.getenv("DELTA_STEP"))

class Concierge(concierge_pb2_grpc.ConciergeServicer):

    def __init__(self) -> None:

        self.apps: Dict[str, PipelineRepr] = {}
        self.request_executor: ThreadPoolExecutor =\
            ThreadPoolExecutor(max_workers=10)
        self.p_executor: ProcessPoolExecutor =\
            ProcessPoolExecutor(max_workers=4)

        self.checkin_lock: Lock = Lock()
        self.request_lock: Lock = Lock()
        self.profiling_lock: Lock = Lock()
        self.baseline_bw_lock: Lock = Lock()

        # bw
        self.next_handle: int = 20
        self.remain_bw: float = MAX_BW
        self.__init_tc()
        self.victim = None
        self.victim_location = None
        self.highest = None
        self.highest_location = None
        self.sensitivity_threshold = 0 # No threshold

        # cpu
        self.remain_cpu: int = MAX_CPUSHARE
        # self.__init_cgroups()

        # gpu
        self.gpu_queue_lock: Lock = Lock()
        self.gpu_queue: SimpleQueue[PipelineRepr] = SimpleQueue()
        self.is_gpu_busy: Event = Event()
        Thread(target=self.__gpu_sched).start()

        # starting reallocation worker
        workers_type: List[ResourceType] = [ResourceType.BW]
        workers = [Thread(target=self.__reallocation_worker,
                          args=(resource_t,))
                   for resource_t in workers_type]
        [t.start() for t in workers]

        # init logger
        self.logger = logging.getLogger("Concierge")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler("./concierge.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-6s %(name)-14s %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        self.logger.info("Initialized")

    # grpc service
    def CheckinApp(self, request, _) -> concierge_pb2.Status:
        self.checkin_lock.acquire()
        default_bw = int(MAX_BW / (len(self.apps) + 1))
        default_cpu =\
            int(MAX_CPUSHARE * (1 - PROFILER_CPU) / (len(self.apps) + 1))
        self.server_port = request.server_port
        new_app = PipelineRepr(
            request.container_id, request.server_uri, request.client_uri, request.server_port,
            self.next_handle, default_bw,
            request.pid, default_cpu, request.is_faked)
        self.apps[request.container_id] = new_app
        self.next_handle += 10

        # distribute the resource evenly when new app checked in
        for _, app in self.apps.items():
            if app != new_app:
                app.tc(MAX_BW / len(self.apps), False, False)
                app.notify_reallocation(True, ResourceType.BW, MAX_BW / len(self.apps))
                # app.cgroups_cpu(default_cpu, False)
                app.notify_reallocation(True, ResourceType.CPU)
                self.remain_bw = 0
                self.remain_cpu = 0
        self.checkin_lock.release()

        self.logger.info(f"App from {request.server_uri} checked in")
        return concierge_pb2.Status(is_succeed=True)

    def ReadyToProfile(self, request, _) -> concierge_pb2.Status:
        app = self.apps[request.container_id]
        resource_t: ResourceType = ResourceType[request.resource_t]
        delta_limit: float = -1
        can_proceed: bool = True

        if resource_t == ResourceType.BW:
            delta_limit = PROFILING_DELTA
            app.tc(delta_limit, True, True)
            can_proceed = app.notify_reallocation(True, ResourceType.BW, PROFILING_DELTA)
        elif resource_t == ResourceType.CPU:
            delta_limit = PROFILING_DELTA
            app.cgroups_cpu(round(delta_limit), True)
            can_proceed = app.notify_reallocation(True, ResourceType.CPU)

        app.start_profiling(delta_limit, can_proceed)
        return concierge_pb2.Status(is_succeed=True)

    def SubmitGPUJob(self, request, _):
        self.gpu_queue_lock.acquire()
        app: PipelineRepr = self.apps[request.container_id]
        self.gpu_queue.put(app)
        self.gpu_queue_lock.release()
        return concierge_pb2.Status(is_succeed=True)

    def DoneGPUJob(self, _, __):
        self.is_gpu_busy.clear()
        return concierge_pb2.Status(is_succeed=True)
    
    # experimental
    def BaselineBW(self, request, _):
        self.baseline_bw_lock.acquire()
        self.apps[request.container_id].f1_scores = request.f1_scores
        if all(app.f1_scores != [] for app in self.apps.values()):
            self.determine_bw()
        self.baseline_bw_lock.release()
        return concierge_pb2.Status(is_succeed=True)

    
    # f1_scores_checker runs in the background
    # hardcoded at 400kbps
    def determine_bw(self):
        # combs = [(400,400,400), (300,400,500), (200,400,600), (500,400,300), (600,400,200), (400,300,500), (400,200,600), (400,500,300), (400,600,200), (300, 500, 400), (200, 600, 400), (500, 300, 400), (600, 200, 400)]
        baseBW = MAX_BW//len(self.apps)
        delta = DELTA_STEP
        # Hardcoded 3 apps
        # PROFILING_DELTA_1 = 50
        combs1 = [(baseBW, baseBW-step, baseBW+step) for step in range(-PROFILING_DELTA, PROFILING_DELTA+delta, delta)]
        combs2 = [(baseBW-step, baseBW, baseBW+step) for step in range(-PROFILING_DELTA, PROFILING_DELTA+delta, delta)]
        combs3 = [(baseBW-step, baseBW+step, baseBW) for step in range(-PROFILING_DELTA, PROFILING_DELTA+delta, delta)]
        combs = [(baseBW, baseBW, baseBW)]
        combs += combs1
        combs += combs2
        combs += combs3

        f1_comb = []
        # writeResult(1, combs, "f1Debugger")
        for comb in combs:
            #f1_comb.append(sum(f1_score_10[comb[0]//100 - 2]+f1_score_10_jakarta[comb[1]//100 -2]+f1_score_10_highway[comb[2]//100 -2])
            f1_comb.append(sum([app[1].f1_scores[(comb[app[0]]-(baseBW-PROFILING_DELTA))//25] for app in enumerate(self.apps.values())]))
        writeResult(1, f1_comb, "f1Debugger")
        best_comb = combs[f1_comb.index(max(f1_comb))]
        writeResult(1, best_comb, "baseline_debugger")
        for app in enumerate(self.apps.values()):
            is_succeed = app[1].tc(best_comb[app[0]], False, False)
        resource_t = ResourceType.BW
        for app in enumerate(self.apps.values()):
            self.request_executor.submit(
                app[1].notify_reallocation, True, resource_t, best_comb[app[0]], True)
        

    def DoneInference(self, request, _) -> concierge_pb2.Status:
        app_id = request.container_id
        for id in self.apps:
            # notify other apps to end inference
            if id != app_id:
                self.apps[id].end_inference()
                self.logger.warning(f"App {id} notified to end inference")

        return concierge_pb2.Status(is_succeed=True)

    def __gpu_sched(self):
        while True:
            # wait for any existing gpu job to finish
            while self.is_gpu_busy.is_set():
                pass
            job: PipelineRepr = self.gpu_queue.get()
            self.is_gpu_busy.set()
            job.start_gpu_job()

    @staticmethod
    def __init_cgroups():
        subprocess.run([*CG_CLR])

    @staticmethod
    def __init_tc():
        # insert ifb kernel module and init the interface ifb0
        subprocess.run(INSERT_KMODULE).check_returncode()
        subprocess.run(IFB_UP).check_returncode()
        # redirect interface IF"s ingress to ifb0"s egress
        # remove the rule first, to handle error
        subprocess.run([*TC_QDISC_DEL, IF, "handle", "ffff:", "ingress"])
        # add new rule
        subprocess.run([*TC_QDISC_ADD, IF, "handle", "ffff:", "ingress"])\
            .check_returncode
        subprocess.run([*TC_FILTER_ADD, IF, "parent", "ffff:",
                        "protocol", "ip", "u32", "match", "u32", "0", "0",
                        "action", "mirred", "egress",
                        "redirect", "dev", "ifb0"]).check_returncode()

        # add root htb class with default class 1:10
        subprocess.run([*TC_QDISC_DEL, IFB, "root", "handle", "1:", "htb"])
        subprocess.run([*TC_QDISC_ADD, IFB, "root", "handle", "1:", "htb",
                        "default", "10"]).check_returncode()
        # total BW class 1:1
        subprocess.run([*TC_CLASS_ADD, IFB, "parent", "1:",
                        "classid", "1:1", "htb",
                        "rate", f"{10* 1024 * 1024}kbit"])\
            .check_returncode()
        # default BW class 1:10
        # do not limit bandwidth if it is not a registered ip source
        subprocess.run([*TC_CLASS_ADD, IFB, "parent", "1:1",
                        "classid", "1:10", "htb",
                        "rate", f"{10 * 1024 * 1024}kbit"])\
            .check_returncode()

    def __reallocation_worker(self, resource_t: ResourceType):
        resource_t_str: str = str(resource_t)[str(resource_t).index(".") + 1:]
        # copy the application(?)
        # the concierge should wait until the total of apps is 3
        while len(self.apps) < NUM_APP:
            pass
        # sanity check
        sleep(15)
        # sleep(24)
        # dds 24 seconds
        while True:
            #  Apps are running
            if(BASELINE_MODE):
                sleep(100000)
            else:
                if MI > 5:
                    sleep(MI)
                else:
                    sleep(5)
            if len(self.apps) < NUM_APP-1:
                continue
            futures: List[Future] = [
                self.request_executor.submit(app.get_diff, resource_t_str)
                for _, app in self.apps.items()]
            # waiting for 20 seconds to do the profiling
            wait(futures, timeout=60)
            requests: List[ResRequest] =\
                [ResRequest(*future.result(), -1) for future in futures]
            
            # get diff periodically called every 20 seconds, i guess
            
            # METHOD 2: Single infer_diff, single sensitivity estimation
            # if any([request.diff.infer_diff == -1 for request in requests]):
            #     self.logger.error(
            #         "Something went wrong when calculating Diff.")
            #     allocation: List[Allocation] = [
            #         Allocation(request.app, True, 0) for request in requests]
            # else:
            #     st = perf_counter()
            #     allocation: List[Allocation] = self.__optimize(requests)
            #     self.logger.debug(f"optimize(): {perf_counter() - st}")

            # METHOD 3: upward sensitivity - the other's downward sensitivity

            # Victim Initialization
            self.total_min_bw = 0
            self.victim = ResRequest(app = -1,
                                        diff = InferDiff(infer_diff = 1,
                                                infer_diff_high = 1,
                                                infer_diff_low = 1,
                                                latency_diff = -1,
                                                bw_lower_bound = -1,
                                                is_min_bw = -1,
                                                curr_frame = -99,
                                                curr_f1 = -1,
                                                curr_budget = 0
                                                ),
                                        amount = -1)
            isVictimFound: bool = False
            self.victim_delta = self.victim.diff.infer_diff_high + self.victim.diff.infer_diff_low

            self.highest = ResRequest(app = -1,
                                        diff = InferDiff(infer_diff = -1,
                                                infer_diff_high = -1,
                                                infer_diff_low = -1,
                                                latency_diff = -1,
                                                bw_lower_bound = -1,
                                                is_min_bw = -1,
                                                curr_frame = -99,
                                                curr_f1 = -1,
                                                curr_budget = 0
                                                ),
                                        amount = -1)
            
            self.lowest_f1 = ResRequest(app = -1,
                                        diff = InferDiff(infer_diff = -1,
                                                infer_diff_high = -1,
                                                infer_diff_low = 1,
                                                latency_diff = -1,
                                                bw_lower_bound = -1,
                                                is_min_bw = -1,
                                                curr_frame = -99,
                                                curr_f1 = 10,
                                                curr_budget = 0
                                                ),
                                        amount = -1)
            # # Find the victim 1.0
            # while not isVictimFound:
            #     for i in range(len(requests)):
            #         if requests[i].diff.infer_diff < self.victim.diff.infer_diff and not requests[i].diff.is_min_bw:
            #             self.victim = requests[i]
            #     if self.victim.app != -1:
            #         isVictimFound = True
            
            # The victim and the min_bw shouldn't be taken account into the lp calculation, 
            # another thing, the min_bw shouldn't be neither added nor reduced by bw, should be discarded, the victim may be reduced
            for i in range(len(requests)):
                infer_diff_low_sum = 0
                for j in range(len(requests)):
                    if i != j:
                        infer_diff_low_sum += requests[j].diff.infer_diff_low
                infer_diff = (requests[i].diff.infer_diff_high - infer_diff_low_sum)
                requests[i] = ResRequest(app = requests[i].app,
                                        diff = InferDiff(infer_diff = infer_diff,
                                                infer_diff_high = requests[i].diff.infer_diff_high,
                                                infer_diff_low = requests[i].diff.infer_diff_low,
                                                latency_diff = requests[i].diff.latency_diff,
                                                bw_lower_bound = requests[i].diff.bw_lower_bound,
                                                is_min_bw = requests[i].diff.is_min_bw,
                                                curr_frame = requests[i].diff.curr_frame,
                                                curr_f1 = requests[i].diff.curr_f1,
                                                curr_budget = requests[i].diff.curr_budget
                                                ),
                                        amount = requests[i].amount) # token stuff
                infer_delta = requests[i].diff.infer_diff_high + requests[i].diff.infer_diff_low
                # Determine the highest

                # Victim determination
                # if requests[i].diff.infer_diff_low < self.victim.diff.infer_diff_low and infer_delta < self.victim_delta and not requests[i].diff.is_min_bw:
                if infer_delta < self.victim_delta and not requests[i].diff.is_min_bw:
                    self.victim_delta = infer_delta
                    self.victim = requests[i]
                    self.victim_location = i
                
                self.total_min_bw = self.total_min_bw + 1 if requests[i].diff.is_min_bw else self.total_min_bw
                
                # if requests[i].diff.curr_f1 < self.lowest_f1.diff.curr_f1:
                #     self.lowest_f1 = requests[i]
                #     self.lowest_f1_location = i

            #Debug
                # self.logger.debug(f"app{i} upon receiving: {requests[i].diff.diff_}, {requests[i].diff.infer_diff_high}, {requests[i].diff.infer_diff_low}, {requests[i].diff.bw_lower_bound} at frame: {requests[i].diff.curr_frame} -> {requests[i].diff.is_min_bw}")
                self.logger.debug(f"app{i}: {requests[i].diff.infer_diff}, {requests[i].diff.infer_diff_high}, {requests[i].diff.infer_diff_low}, {requests[i].diff.bw_lower_bound} at frame: {requests[i].diff.curr_frame} -> {requests[i].diff.is_min_bw}")
            #DEBUG, the larger the parameter, the more likely to be allocated

            # # Recovery data for other request, only respect to the victim, except the min_bw one:
            for i in range(len(requests)):
                if requests[i].app != self.victim.app:
                    temp = requests[i].diff.infer_diff_high - self.victim.diff.infer_diff_low
                    diffTemp = requests[i].diff._replace(infer_diff=temp)
                    requests[i] = requests[i]._replace(diff=diffTemp)

            # # find the highest sensitivity
            # for i in range(len(requests)):
            #     if requests[i].app != self.victim.app and requests[i].diff.infer_diff > self.highest.diff.infer_diff:
            #         self.highest = requests[i]
            #         self.highest_location = i

            # # correction
            # for i in range(len(requests)):
            #     if requests[i].diff.is_min_bw and requests[i].app != self.highest.app:
            #         temp = self.victim.diff.infer_diff+0.005
            #         diffTemp = requests[i].diff._replace(infer_diff=temp)
            #         requests[i] = requests[i]._replace(diff=diffTemp)
           
            # diffTemp = requests[self.victim_location].diff._replace(infer_diff=0)
            # requests[self.victim_location] = requests[self.victim_location]._replace(diff=diffTemp)

            if self.victim.app != -1:
                self.logger.debug(f"The victim of this iteration is {self.victim.app.container_id}")
            else:
                self.logger.debug(f"The victim of this iteration is none")
            # End of recovery data process

            client_uris = np.array([request.app.container_id for request in requests])
            normalized_infer_gradients = np.array([request.diff.infer_diff for request in requests])
            self.logger.info(client_uris)
            self.logger.info(normalized_infer_gradients)

            # if all([request.diff.infer_diff <= 0 for request in requests]) or self.total_min_bw == len(requests)-1:
            if all([request.diff.infer_diff <= 0 for request in requests]):
                # pass
                # st = perf_counter()
                # allocation: List[Allocation] = self.__optimize_negative(requests)
                # self.logger.debug(f"optimize(): {perf_counter() - st}")

                self.logger.error(
                    "All upward sensitivity is smaller than or equal to the sum of the other's downward sensitivity. Keep the current allocation.")
                allocation: List[Allocation] = [
                    Allocation(request.app, True, 0) for request in requests]
            else:
                 # if the sensitivity all the same, the one with the smallest curr_f1 wins        
                # if all([request.diff.infer_diff == self.victim.diff.infer_diff for request in requests]):
                #     temp = requests[self.lowest_f1_location].diff.infer_diff + 0.01
                #     diffTemp = requests[self.lowest_f1_location].diff._replace(infer_diff=temp)
                #     requests[self.lowest_f1_location] = requests[self.lowest_f1_location]._replace(diff=diffTemp)
                st = perf_counter()
                allocation: List[Allocation] = self.__optimize(requests)
                self.logger.debug(f"optimize(): {perf_counter() - st}")

            st = perf_counter()
            self.__reallocate(resource_t, allocation)
            self.logger.debug(f"reallocate(): {perf_counter() - st}")
            if MI >= 5:
                sleep(MI-5)
            else:
                pass
    
    def __optimize_negative(self, requests: List[ResRequest]) -> List[Allocation]:
        allocations = [0 for i in range(len(requests))]
        res: List[Allocation] = []
        if self.victim.app != self.highest.app:
            # if infer_diff_high is 0, we shouldn't take from the victim
            if self.victim.diff.infer_diff_low == 0 and self.highest.diff.infer_diff_high != 0:
                # allocate the rest to the highest sensitivity app
                allocationVal = min(PROFILING_DELTA,
                self.victim.app.current_bw - self.victim.diff.bw_lower_bound)
                allocations[self.highest_location] = allocationVal
                allocations[self.victim_location] = -1*allocationVal
        for i in range(len(requests)):
            res.append(Allocation(
                requests[i].app,
                requests[i].app.current_bw + allocations[i]
                    >= requests[i].diff.bw_lower_bound,
                allocations[i]))
        return res


    def __optimize(self, requests: List[ResRequest]) -> List[Allocation]:
        # LP
        # Find a vector dBW
        # Maximize:     (infer_gradient + lat_gradient)^T * dRes
        # Subject to:   sum(request_dBW_i) - sum(non_request_dBW_i) <= 0,
        #               and for all i in app_lst
        #               -current_BW_i <= dBW_i <= total_BW if app_i did not
        #               request

        def normalize(arr):
            max = np.amax(arr)
            min = np.amin(arr)
            if max == 0:
                return arr
            if max - min == 0:
                return arr / max
            return (arr - min) / (max - min)

        # normalize latency gradients and accuracy gradients to the same scale
        # using min-max normalization for now
        normalized_infer_gradients = np.array(
            [request.diff.infer_diff  for request in requests])
        res: List[Allocation] = []
        if all([gradient < self.sensitivity_threshold for gradient in normalized_infer_gradients]):
            self.logger.info(f"Sensitivities are all zero or under the threshold ({self.sensitivity_threshold}).")
            for i in range(len(requests)):
                res.append(Allocation(requests[i].app, True, 0))
            return res
 
        obj = normalized_infer_gradients * -1

        conservation_lhs: np.ndarray = np.array(
                [[1 for request in requests]])
        conservation_rhs: np.ndarray = np.array([0])
        boundaries: np.ndarray = np.array(
            # [(max(-request.app.current_bw * PROFILING_DELTA,
            [(max(-abs(PROFILING_DELTA),
                  (request.diff.bw_lower_bound) - request.app.current_bw) if request.app == self.victim.app else 0, # the given min_bw or the bw lower bound of the lastest segment
              # request.app.current_bw * PROFILING_DELTA)
              0 if (request.app == self.victim.app) else abs(PROFILING_DELTA)) 
             for request in requests])
        # self.logger.info(boundaries)
        lp_res: op.OptimizeResult = self.p_executor.submit(
            op.linprog, obj, A_eq=conservation_lhs, b_eq=conservation_rhs,
            bounds=boundaries).result()
        is_succeed: bool = lp_res.success
        res: List[Allocation] = []
        for i in range(len(requests)):
            res.append(Allocation(
                requests[i].app,
                (is_succeed and requests[i].app.current_bw + lp_res.x[i]
                    >= (requests[i].diff.bw_lower_bound)),
                lp_res.x[i]
                if lp_res.success else 0))
        return res
# 
    def __reallocate(self, resource_t: ResourceType,
                     allocations: List[Allocation]):
        # pass if cannot find a solution from _optimize()
        if len(allocations) == 0:
            return

        final_res: List[Allocation] = []
        # allocation.amount will be negative if should be substracted and will be positive if should be added
        for allocation in allocations:
            app: PipelineRepr = allocation.app
            if resource_t == ResourceType.BW:
                is_succeed = app.tc(allocation.amount, True, False)
                # Consistency calculation, allocation = 0
                # is_succeed = app.tc(0, True, False)
                self.remain_bw -= allocation.amount
            elif resource_t == ResourceType.CPU:
                is_succeed = app.cgroups_cpu(int(allocation.amount), True)
                self.remain_cpu += int(allocation.amount)
            else:
                is_succeed = False
            final_res.append(
                Allocation(app, is_succeed, allocation.amount))

        self.logger.debug(final_res)
        for allocation in final_res:
            self.request_executor.submit(
                allocation[0].notify_reallocation, allocation[1], resource_t, allocation[2])

            # # # Consistency calculation
            # self.request_executor.submit(
            #     allocation[0].notify_reallocation, allocation[1], resource_t, 0)


def serve(servicer: concierge_pb2_grpc.ConciergeServicer):
    server = grpc.server(ThreadPoolExecutor(20))
    concierge_pb2_grpc.add_ConciergeServicer_to_server(servicer, server)
    server.add_insecure_port("0.0.0.0:5000")
    server.start()
    server.wait_for_termination()

def writeResult(appNum, latency, file_path) -> any:
    f = open(f"../../{file_path}-{appNum}.csv", "a")
    f.write(str(latency) + '\n')
    f.close()


if __name__ == "__main__":
    serve(Concierge())
