import os
import subprocess
from threading import Thread
from time import sleep
from typing import NamedTuple, Tuple

import logging

import grpc
from google.protobuf.empty_pb2 import Empty

from api.common.command import (
    CG_CREATE,
    CG_MOV,
    CG_SET,
    CPU_SHR,
    IFB,
    TC_CLASS_REPLACE,
    TC_FILTER_ADD,
    TC_CLASS_ADD,
    TC_FILTER_REPLACE,
)
from api.common.enums import ResourceType
import api.proto.app_server_pb2 as pb2
import api.proto.app_server_pb2_grpc as pb2_grpc
# from app.ddsdds_utils import writeResults
from dds_utils import writeResult

MIN_BW = int(os.getenv("MIN_BW") or "20000")


class InferDiff(NamedTuple):
    infer_diff: float
    latency_diff: float
    # METHOD 3: [online] 1 - old-to-new f1, both directions
    infer_diff_high: float
    infer_diff_low: float
    bw_lower_bound: float
    is_min_bw: bool
    curr_frame: int
    curr_f1: float
    curr_budget: float


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PipelineRepr():

    def __init__(self, container_id: str, server_uri: str, client_uri: str, data_port: int,
                 tc_class_id: int, default_bw: int,
                 pid: int, default_cpushare: int, is_faked: bool):
        
        self.logger = logging.getLogger("PipelineRepr")
        format = logging.Formatter("[%(name)s][%(levelname)s] - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(format)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.container_id: str = container_id
        self.server_uri: str = server_uri
        self.client_uri: str = client_uri
        self.pid: str = str(pid)
        self.min_bw = MIN_BW
        self.is_faked = is_faked

        channel = grpc.insecure_channel(self.server_uri)
        self.stub = pb2_grpc.PipelineStub(channel)

        # bandwidth
        self.tc_class_id = tc_class_id
        self.current_bw: int = default_bw
        self.last_bw_change = default_bw
        self.last_bw_change_profiling = False
        self.data_port = data_port
        self.__add_tc_class()
        
        # new baseline framework
        self.f1_scores = []
        self.curr_budget = 0.45

        # cpu
        self.current_cpu: int = default_cpushare
        self.last_cpu_change = self.current_cpu
        # self.__create_cgroup()

        # gpu
        self.refill_interval: float = 0.5
        self.refill_amount: int = 500_000_000
        Thread(target=self.__refill_worker).start()

    def __refill_worker(self):
        sleep(self.refill_interval)
        while True:
            self.stub.RefillToken(pb2.NewTokens(amount=self.refill_amount))
            sleep(self.refill_interval)

    def __create_cgroup(self):
        cgroup = f"cpu:/{self.container_id}"
        subprocess.run([*CG_CREATE, cgroup])
        subprocess.run([*CG_MOV, cgroup, self.pid])

    def __add_tc_class(self):
        subprocess.run([*TC_CLASS_ADD, IFB, "parent", "1:1",
                        "classid", f"1:{self.tc_class_id}",
                        "htb", "rate", f"{self.current_bw if not self.is_faked else self.current_bw - 250}kbit"])
        subprocess.run([*TC_FILTER_ADD, IFB, "protocol", "ip", "parent", "1:", "prio", "1",
                        "u32", "match", "ip", "src", f"{self.client_uri}",
                        "match", "ip", "dport", f"{self.data_port}", "0xffff",
                        "flowid", f"1:{self.tc_class_id}"])
        self.notify_reallocation(True, ResourceType.BW)

    def notify_reallocation(self, is_succeed: bool, resource_t: ResourceType, amount=None, clear_baseline=False)\
            -> bool:
        resource_t_str = str(resource_t)
        resource_change = -1
        # comes from the profiling or checkin
        if amount == None:
            if resource_t == ResourceType.BW:
                resource_change = self.last_bw_change
            elif resource_t == ResourceType.CPU:
                resource_change = self.last_cpu_change
        else:
            resource_change = amount
        notification = {
                "is_succeed": is_succeed,
                "actual_amount": resource_change,
                "resource_t": resource_t_str[resource_t_str.index(".") + 1:],
                "can_clear_profiling_flag": not self.last_bw_change_profiling,
                "is_min_bw": (self.min_bw) >= self.current_bw if not self.last_bw_change_profiling else False,
                "can_proceed": clear_baseline,
                }
        response: bool = self.stub.NotifyReallocation(
                pb2.Notification(**notification)).notified
        return response

    def start_profiling(self, resource_change: float, can_proceed: bool):
        self.stub.StartProfile(pb2.ResourceChange(
            change=resource_change, can_proceed=can_proceed))

    def get_diff(self, resource_t_str) -> Tuple["PipelineRepr", InferDiff]:
        response = self.stub.GetDiff(
                pb2.ResourceType(resource_t=resource_t_str))
        
        # self.logger.warning(f"GetDiff response: {response}")

        # return self, InferDiff(response.infer_diff, response.latency_diff)
        # self.min_bw = response.bw_lower_bound
        # self.curr_budget = response.curr_budget
        # test = InferDiff(response.infer_diff, response.latency_diff, response.infer_diff_high, response.infer_diff_low,
        #  response.bw_lower_bound, response.is_min_bw, response.curr_frame, response.curr_f1, response.curr_budget)
        # writeResult(self.client_uri, test, "messageDebug")
        # writeResults(self.container_id, [response.infer_diff_high, response.infer_diff_low], "conciergeDebug")
        return self, InferDiff(response.infer_diff, response.latency_diff, response.infer_diff_high, response.infer_diff_low,
                            response.bw_lower_bound, response.is_min_bw, response.curr_frame, response.curr_f1, response.curr_budget)

    def cgroups_cpu(self, new_share: int, is_relative: bool) -> bool:
        prev_cpu = self.current_cpu
        if is_relative:
            self.current_cpu += new_share
        else:
            self.current_cpu = new_share
        self.last_cpu_change = self.current_cpu - prev_cpu
        subprocess.run([*CG_SET, f"{CPU_SHR}={self.current_cpu}",
                        self.container_id])
        return True

    def tc(self, rate: float, is_relative: bool, is_profiling: bool) -> bool:
        # new_limit = self.current_bw + ((1/self.curr_budget) * round(rate))\
        #     if is_relative else round(rate)
        new_limit = self.current_bw + round(rate)\
            if is_relative else round(rate)
        subprocess.run([*TC_CLASS_REPLACE, IFB, "parent", "1:1",
                        "classid", f"1:{self.tc_class_id}",
                        "htb", "rate", f"{new_limit if not self.is_faked else new_limit - 250}kbit"])\
            .check_returncode()
        subprocess.run([*TC_FILTER_REPLACE, IFB, "protocol", "ip", "parent", "1:", "prio", "1",
                        "u32", "match", "ip", "src", f"{self.client_uri}",
                        "match", "ip", "dport", f"{self.data_port}", "0xffff",
                        "flowid", f"1:{self.tc_class_id}"])
        # last change from profiling (?)
        if self.last_bw_change_profiling:
            self.last_bw_change =\
                new_limit - self.current_bw - self.last_bw_change 
        # slbc = 830 - 750 - 80 = 0 ?, if down 670 - 750 - 80
        else:
            self.last_bw_change = new_limit - self.current_bw

        if not is_profiling:
            self.current_bw = new_limit
            self.last_bw_change_profiling = False # last_bw_change after profiling
        else:
            self.last_bw_change_profiling = True

        ## Personal Note:
        # pas awal profiling, new_limit = 830, slbc = 80, scb = 830, slbcp = false, trs pas downward, tar, new_limit = 670, slbc = 670 - 830 = -160, slbcp = true, tar kalo
        # udh dialokasi, misal new_limit = 830, slbc = 830 - 830 + 160 = 160

        return True

    def start_gpu_job(self):
        self.stub.StartGPUJob(Empty())

    def end_inference(self):
        self.stub.EndInference(Empty())
