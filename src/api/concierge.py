import logging
from concurrent.futures import Future, ThreadPoolExecutor, wait
from concurrent.futures.process import ProcessPoolExecutor
import os
import sys
from queue import SimpleQueue
import subprocess
from threading import Event, Lock, Thread
from time import sleep, perf_counter_ns as perf_counter
from typing import Dict, List

import grpc
import numpy as np
from scipy import optimize as op

from api.common.command import (
    CG_CLR, IF, IFB, IFB_UP, INSERT_KMODULE, TC_CLASS_REPLACE, TC_CLASS_ADD,
    TC_FILTER_REPLACE, TC_FILTER_ADD, TC_QDISC_ADD, TC_QDISC_DEL,
)
from api.common.enums import ResourceType
from api.common.typedef import Allocation, ResRequest
from api.pipeline_repr import PipelineRepr, InferDiff
import api.proto.concierge_pb2 as concierge_pb2
import api.proto.concierge_pb2_grpc as concierge_pb2_grpc

# Global Logger Setup
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-6s %(name)-14s %(message)s',
                    handlers=[
                        logging.FileHandler("./global_concierge.log"),
                        logging.StreamHandler()
                    ])

global_logger = logging.getLogger("GlobalConcierge")

PROFILER_CPU: float = float(os.getenv("PROFILER_CPU") or 0.1)
PROFILER_MEM: float = float(os.getenv("PROFILER_MEM") or 1)
PROFILER_CG: str = os.getenv("PROFILER_CG") or "profiler"

MAX_BW: int = int(os.getenv("MAX_BW") or 60 * 1024)
MI: int = int(os.getenv("MI") or 5)
MAX_CPUSHARE: int = int(os.getenv("MAX_CPUSHARE") or 1000)
PROFILING_DELTA: int = int(os.getenv("PROFILING_DELTA") or 5000)
BASELINE_MODE: bool = int(os.getenv("BASELINE_MODE")) > 0
NUM_APP: int = int(os.getenv("NUM_APP") or 1)
DELTA_STEP: int = int(os.getenv("DELTA_STEP"))

class Concierge(concierge_pb2_grpc.ConciergeServicer):

    def __init__(self) -> None:
        try:
            self.apps: Dict[str, PipelineRepr] = {}
            self.request_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=10)
            self.p_executor: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=4)

            self.checkin_lock: Lock = Lock()
            self.request_lock: Lock = Lock()
            self.profiling_lock: Lock = Lock()
            self.baseline_bw_lock: Lock = Lock()

            # Bandwidth and CPU initialization
            self.next_handle: int = 20
            self.remain_bw: float = MAX_BW
            self.__init_tc()
            self.victim = None
            self.victim_location = None
            self.highest = None
            self.highest_location = None
            self.sensitivity_threshold = 0  # No threshold

            self.remain_cpu: int = MAX_CPUSHARE

            # GPU queue
            self.gpu_queue_lock: Lock = Lock()
            self.gpu_queue: SimpleQueue[PipelineRepr] = SimpleQueue()
            self.is_gpu_busy: Event = Event()
            Thread(target=self.__gpu_sched).start()

            # Starting reallocation worker
            workers_type: List[ResourceType] = [ResourceType.BW]
            workers = [Thread(target=self.__reallocation_worker, args=(resource_t,))
                       for resource_t in workers_type]
            [t.start() for t in workers]
            # Instance-specific logger
            self.logger = logging.getLogger(f"Concierge-{id(self)}")

            file_handler = logging.FileHandler(f"./concierge_{id(self)}.log")
            file_handler.setLevel(logging.DEBUG)

            # Log to stdout for tee
            # stream_handler = logging.StreamHandler(sys.stdout)  
            # stream_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter("%(asctime)s %(levelname)-6s %(name)-14s %(message)s")
            file_handler.setFormatter(formatter)
            # stream_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            # self.logger.addHandler(stream_handler)
            self.logger.setLevel(logging.DEBUG)

            self.logger.info("Initialized Concierge service")
        except Exception as e:
            global_logger.error(f"Error during Concierge initialization: {e}")

    def CheckinApp(self, request, _) -> concierge_pb2.Status:
        try:
            self.checkin_lock.acquire()
            default_bw = int(MAX_BW / (len(self.apps) + 1))
            default_cpu = int(MAX_CPUSHARE * (1 - PROFILER_CPU) / (len(self.apps) + 1))
            self.server_port = request.server_port
            new_app = PipelineRepr(
                request.container_id, request.server_uri, request.client_uri, request.server_port,
                self.next_handle, default_bw,
                request.pid, default_cpu, request.is_faked)
            self.apps[request.container_id] = new_app
            self.next_handle += 10

            # Distribute resources evenly when a new app checks in
            for _, app in self.apps.items():
                if app != new_app:
                    app.tc(MAX_BW / len(self.apps), False, False)
                    app.notify_reallocation(True, ResourceType.BW, MAX_BW / len(self.apps))
                    app.notify_reallocation(True, ResourceType.CPU)
                    self.remain_bw = 0
                    self.remain_cpu = 0
            self.logger.info("App from %s checked in", request.server_uri)
            return concierge_pb2.Status(is_succeed=True)
        except Exception as e:
            self.logger.error("Error during CheckinApp: %s", e)
            return concierge_pb2.Status(is_succeed=False)
        finally:
            self.checkin_lock.release()

    def ReadyToProfile(self, request, _) -> concierge_pb2.Status:
        try:
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
        except Exception as e:
            self.logger.error("Error during ReadyToProfile: %s", e)
            return concierge_pb2.Status(is_succeed=False)

    def SubmitGPUJob(self, request, _):
        try:
            self.gpu_queue_lock.acquire()
            app: PipelineRepr = self.apps[request.container_id]
            self.gpu_queue.put(app)
            return concierge_pb2.Status(is_succeed=True)
        except Exception as e:
            self.logger.error("Error during SubmitGPUJob: %s", e)
            return concierge_pb2.Status(is_succeed=False)
        finally:
            self.gpu_queue_lock.release()

    def DoneGPUJob(self, _, __):
        try:
            self.is_gpu_busy.clear()
            return concierge_pb2.Status(is_succeed=True)
        except Exception as e:
            self.logger.error("Error during DoneGPUJob: %s", e)
            return concierge_pb2.Status(is_succeed=False)

    def BaselineBW(self, request, _):
        try:
            self.baseline_bw_lock.acquire()
            self.apps[request.container_id].f1_scores = request.f1_scores
            if all(app.f1_scores != [] for app in self.apps.values()):
                self.determine_bw()
            return concierge_pb2.Status(is_succeed=True)
        except Exception as e:
            self.logger.error("Error during BaselineBW: %s", e)
            return concierge_pb2.Status(is_succeed=False)
        finally:
            self.baseline_bw_lock.release()

    def determine_bw(self):
        try:
            baseBW = MAX_BW // len(self.apps)
            delta = DELTA_STEP

            combs1 = [(baseBW, baseBW - step, baseBW + step)
                      for step in range(-PROFILING_DELTA, PROFILING_DELTA + delta, delta)]
            combs2 = [(baseBW - step, baseBW, baseBW + step)
                      for step in range(-PROFILING_DELTA, PROFILING_DELTA + delta, delta)]
            combs3 = [(baseBW - step, baseBW + step, baseBW)
                      for step in range(-PROFILING_DELTA, PROFILING_DELTA + delta, delta)]
            combs = [(baseBW, baseBW, baseBW)]
            combs += combs1
            combs += combs2
            combs += combs3

            f1_comb = []
            for comb in combs:
                f1_comb.append(sum([app[1].f1_scores[(comb[app[0]] - (baseBW - PROFILING_DELTA)) // 25]
                                    for app in enumerate(self.apps.values())]))
            writeResult(1, f1_comb, "f1Debugger")
            best_comb = combs[f1_comb.index(max(f1_comb))]
            writeResult(1, best_comb, "baseline_debugger")
            for app in enumerate(self.apps.values()):
                app[1].tc(best_comb[app[0]], False, False)
            resource_t = ResourceType.BW
            for app in enumerate(self.apps.values()):
                self.request_executor.submit(
                    app[1].notify_reallocation, True, resource_t, best_comb[app[0]], True)
        except Exception as e:
            self.logger.error("Error during determine_bw: %s", e)

    def DoneInference(self, request, _) -> concierge_pb2.Status:
        try:
            app_id = request.container_id
            for id in self.apps:
                if id != app_id:
                    self.apps[id].end_inference()
                    self.logger.warning("App %s notified to end inference", id)
            return concierge_pb2.Status(is_succeed=True)
        except Exception as e:
            self.logger.error("Error during DoneInference: %s", e)
            return concierge_pb2.Status(is_succeed=False)

    def __gpu_sched(self):
        try:
            while True:
                while self.is_gpu_busy.is_set():
                    pass
                job: PipelineRepr = self.gpu_queue.get()
                self.is_gpu_busy.set()
                job.start_gpu_job()
        except Exception as e:
            self.logger.error("Error during GPU scheduling: %s", e)

    @staticmethod
    def __init_cgroups():
        try:
            subprocess.run([*CG_CLR])
        except Exception as e:
            global_logger.error("Error during cgroups initialization: %s", e)

    @staticmethod
    def __run_command_with_logging(command, success_message, failure_message):
        try:
            subprocess.run(command, check=True)
            global_logger.info(success_message)
        except subprocess.CalledProcessError as e:
            global_logger.error(f"{failure_message}: {e}")
            global_logger.error("Command: %s, Return code: %s, Output: %s", e.cmd, e.returncode, e.output)

    @staticmethod
    def __init_tc():
        commands = [
            (INSERT_KMODULE, "IFB kernel module inserted", "Failed to insert IFB kernel module"),
            (IFB_UP, "IFB interface brought up", "Failed to bring up IFB interface"),
            ([*TC_QDISC_DEL, IF, "handle", "ffff:", "ingress"], "Existing qdisc rule deleted", "Failed to delete existing qdisc rule"),
            ([*TC_QDISC_ADD, IF, "handle", "ffff:", "ingress"], "New qdisc rule added", "Failed to add new qdisc rule"),
            ([*TC_FILTER_ADD, IF, "parent", "ffff:",
            "protocol", "ip", "u32", "match", "u32", "0", "0",
            "action", "mirred", "egress",
            "redirect", "dev", "ifb0"], "Filter rule for redirect added", "Failed to add filter rule for redirect"),
            ([*TC_QDISC_DEL, IFB, "root", "handle", "1:", "htb"], "Existing root htb class deleted", "Failed to delete existing root htb class"),
            ([*TC_QDISC_ADD, IFB, "root", "handle", "1:", "htb", "default", "10"], "Root htb class added", "Failed to add root htb class"),
            ([*TC_CLASS_ADD, IFB, "parent", "1:", "classid", "1:1", "htb", "rate", f"{10 * 1024 * 1024}kbit"], "Total bandwidth class added", "Failed to add total bandwidth class"),
            ([*TC_CLASS_ADD, IFB, "parent", "1:1", "classid", "1:10", "htb", "rate", f"{10 * 1024 * 1024}kbit"], "Default bandwidth class added", "Failed to add default bandwidth class")
        ]

        for command, success_msg, failure_msg in commands:
            Concierge.__run_command_with_logging(command, success_msg, failure_msg)

        global_logger.info("Traffic control initialization completed successfully.")
    def __reallocation_worker(self, resource_t: ResourceType):
        try:
            resource_t_str: str = str(resource_t)[str(resource_t).index(".") + 1:]
            while len(self.apps) < NUM_APP:
                pass
            sleep(15)

            while True:
                if BASELINE_MODE:
                    sleep(100000)
                else:
                    sleep(max(MI, 5))
                if len(self.apps) < NUM_APP-1:
                    continue

                futures: List[Future] = [
                    self.request_executor.submit(app.get_diff, resource_t_str)
                    for _, app in self.apps.items()]
                wait(futures, timeout=60)
                requests: List[ResRequest] = [ResRequest(*future.result(), -1) for future in futures]
                
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
                                            amount = requests[i].amount)
                    infer_delta = requests[i].diff.infer_diff_high + requests[i].diff.infer_diff_low

                    if infer_delta < self.victim_delta and not requests[i].diff.is_min_bw:
                        self.victim_delta = infer_delta
                        self.victim = requests[i]
                        self.victim_location = i
                    
                    self.total_min_bw = self.total_min_bw + 1 if requests[i].diff.is_min_bw else self.total_min_bw

                    self.logger.debug("app%s: %s, %s, %s, %s at frame: %s -> %s", 
                                      i, 
                                      requests[i].diff.infer_diff, 
                                      requests[i].diff.infer_diff_high, 
                                      requests[i].diff.infer_diff_low, 
                                      requests[i].diff.bw_lower_bound, 
                                      requests[i].diff.curr_frame, 
                                      requests[i].diff.is_min_bw)

                for i in range(len(requests)):
                    if requests[i].app != self.victim.app:
                        temp = requests[i].diff.infer_diff_high - self.victim.diff.infer_diff_low
                        diffTemp = requests[i].diff._replace(infer_diff=temp)
                        requests[i] = requests[i]._replace(diff=diffTemp)

                if self.victim.app != -1:
                    self.logger.debug("The victim of this iteration is %s", self.victim.app.container_id)
                else:
                    self.logger.debug("The victim of this iteration is none")

                client_uris = np.array([request.app.container_id for request in requests])
                normalized_infer_gradients = np.array([request.diff.infer_diff for request in requests])
                self.logger.info(client_uris)
                self.logger.info(normalized_infer_gradients)

                if all([request.diff.infer_diff <= 0 for request in requests]):
                    self.logger.debug("optimize(): %s", perf_counter() - st)

                    self.logger.error(
                        "All upward sensitivity is smaller than or equal to the sum of the other's downward sensitivity. Keep the current allocation.")
                    allocation: List[Allocation] = [
                        Allocation(request.app, True, 0) for request in requests]
                else:
                    st = perf_counter()
                    allocation: List[Allocation] = self.__optimize(requests)
                    self.logger.debug("optimize(): %s", perf_counter() - st)

                st = perf_counter()
                self.__reallocate(resource_t, allocation)
                self.logger.debug("reallocate(): %s", perf_counter() - st)
                if MI >= 5:
                    sleep(MI-5)
                else:
                    pass
        except Exception as e:
            self.logger.error("Error during reallocation worker: %s", e)

    def __optimize_negative(self, requests: List[ResRequest]) -> List[Allocation]:
        try:
            allocations = [0 for i in range(len(requests))]
            res: List[Allocation] = []
            if self.victim.app != self.highest.app:
                if self.victim.diff.infer_diff_low == 0 and self.highest.diff.infer_diff_high != 0:
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
        except Exception as e:
            self.logger.error("Error in optimization: %s", e)


    def __optimize(self, requests: List[ResRequest]) -> List[Allocation]:
        try:
            def normalize(arr):
                max_val = np.amax(arr)
                min_val = np.amin(arr)
                if max_val == 0:
                    return arr
                if max_val - min_val == 0:
                    return arr / max_val
                return (arr - min_val) / (max_val - min_val)

            normalized_infer_gradients = np.array(
                [request.diff.infer_diff for request in requests])
            res: List[Allocation] = []
            if all([gradient < self.sensitivity_threshold for gradient in normalized_infer_gradients]):
                self.logger.info("Sensitivities are all zero or under the threshold (%s).", self.sensitivity_threshold)
                for i in range(len(requests)):
                    res.append(Allocation(requests[i].app, True, 0))
                return res

            obj = normalized_infer_gradients * -1

            conservation_lhs: np.ndarray = np.array([[1 for request in requests]])
            conservation_rhs: np.ndarray = np.array([0])
            boundaries: np.ndarray = np.array(
                [(max(-abs(PROFILING_DELTA),
                      (request.diff.bw_lower_bound) - request.app.current_bw) if request.app == self.victim.app else 0,
                  0 if (request.app == self.victim.app) else abs(PROFILING_DELTA))
                 for request in requests])

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
        except Exception as e:
            self.logger.error("Error in optimization: %s", e)
            return []

    def __reallocate(self, resource_t: ResourceType, allocations: List[Allocation]):
        try:
            if len(allocations) == 0:
                return

            final_res: List[Allocation] = []
            for allocation in allocations:
                app: PipelineRepr = allocation.app
                if resource_t == ResourceType.BW:
                    is_succeed = app.tc(allocation.amount, True, False)
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
        except Exception as e:
            self.logger.error("Error during reallocation: %s", e)

def serve(servicer: concierge_pb2_grpc.ConciergeServicer):
    try:
        server = grpc.server(ThreadPoolExecutor(20))
        concierge_pb2_grpc.add_ConciergeServicer_to_server(servicer, server)
        server.add_insecure_port("0.0.0.0:5000")
        server.start()
        global_logger.info("gRPC server started on port 5000")
        server.wait_for_termination()
    except Exception as e:
        global_logger.error("Error starting gRPC server: %s", e)


def writeResult(appNum, latency, file_path) -> any:
    try:
        with open(f"../../{file_path}-{appNum}.csv", "a") as f:
            f.write(str(latency) + '\n')
    except Exception as e:
        logger.error("Error writing result to file: %s", e)

if __name__ == "__main__":
    try:
        serve(Concierge())
    except Exception as e:
        global_logger.error("Unhandled error in main execution: %s", e)

