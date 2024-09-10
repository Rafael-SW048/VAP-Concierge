import os
import sys
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
from dds_utils import writeResult

# Global logging setup
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-6s %(name)-14s %(message)s',
                    handlers=[
                        logging.FileHandler("./global_pipeline_repr.log"),
                        logging.StreamHandler()
                    ]
)

global_logger = logging.getLogger("GlobalPipelineRepr")
global_logger.setLevel(logging.DEBUG)

MIN_BW = int(os.getenv("MIN_BW") or "20000")

class InferDiff(NamedTuple):
    infer_diff: float
    latency_diff: float
    infer_diff_high: float
    infer_diff_low: float
    bw_lower_bound: float
    is_min_bw: bool
    curr_frame: int
    curr_f1: float
    curr_budget: float


class PipelineRepr:
    
    def __init__(self, container_id: str, server_uri: str, client_uri: str, data_port: int,
                 tc_class_id: int, default_bw: int,
                 pid: int, default_cpushare: int, is_faked: bool):
        
        # Instance-level logger setup
        self.logger = logging.getLogger(f"PipelineRepr-{container_id}")

        file_handler = logging.FileHandler(f"./pipeline_{container_id}.log")
        file_handler.setLevel(logging.DEBUG)

        # stream_handler = logging.StreamHandler(sys.stdout)  # Log to stdout for tee
        # stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(levelname)-6s %(name)-14s %(message)s')
        file_handler.setFormatter(formatter)
        # stream_handler.setFormatter(formatter)

        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(stream_handler)
        
        try:
            self.logger.debug("Initializing PipelineRepr instance")

            self.container_id: str = container_id
            self.server_uri: str = server_uri
            self.client_uri: str = client_uri
            self.pid: str = str(pid)
            self.min_bw = MIN_BW
            self.is_faked = is_faked

            channel = grpc.insecure_channel(self.server_uri)
            self.stub = pb2_grpc.PipelineStub(channel)

            # Bandwidth initialization
            self.tc_class_id = tc_class_id
            self.current_bw: int = default_bw
            self.last_bw_change = default_bw
            self.last_bw_change_profiling = False
            self.data_port = data_port
            self.__add_tc_class()

            # New baseline framework
            self.f1_scores = []
            self.curr_budget = 0.45

            # CPU initialization
            self.current_cpu: int = default_cpushare
            self.last_cpu_change = self.current_cpu

            # GPU token refill mechanism
            self.refill_interval: float = 0.5
            self.refill_amount: int = 500_000_000
            Thread(target=self.__refill_worker).start()

        except Exception as e:
            self.logger.error(f"Error during initialization: {e}", exc_info=True)

    def __refill_worker(self):
        try:
            self.logger.debug("Starting GPU refill worker")
            sleep(self.refill_interval)
            while True:
                self.stub.RefillToken(pb2.NewTokens(amount=self.refill_amount))
                self.logger.debug(f"Refilled GPU tokens with amount: {self.refill_amount}")
                sleep(self.refill_interval)
        except Exception as e:
            self.logger.error(f"Error in refill worker: {e}", exc_info=True)

    def __create_cgroup(self):
        try:
            self.logger.debug("Creating cgroup for container ID: %s", self.container_id)
            cgroup = f"cpu:/{self.container_id}"
            subprocess.run([*CG_CREATE, cgroup])
            subprocess.run([*CG_MOV, cgroup, self.pid])
        except Exception as e:
            self.logger.error(f"Error in creating cgroup: {e}", exc_info=True)

    def __add_tc_class(self):
        try:
            self.logger.debug("Adding TC class for container ID: %s with default bandwidth: %d", self.container_id, self.current_bw)
            
            # Add TC class
            result = subprocess.run(
                [*TC_CLASS_ADD, IFB, "parent", "1:1",
                "classid", f"1:{self.tc_class_id}",
                "htb", "rate", f"{self.current_bw if not self.is_faked else self.current_bw - 250}kbit"],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.debug(f"TC class add command output: {result.stdout}")
            self.logger.debug(f"TC class add command error (if any): {result.stderr}")
            
            # Add TC filter
            result = subprocess.run(
                [*TC_FILTER_ADD, IFB, "protocol", "ip", "parent", "1:", "prio", "1",
                "u32", "match", "ip", "src", f"{self.client_uri}",
                "match", "ip", "dport", f"{self.data_port}", "0xffff",
                "flowid", f"1:{self.tc_class_id}"],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.debug(f"TC filter add command output: {result.stdout}")
            self.logger.debug(f"TC filter add command error (if any): {result.stderr}")

            # Notify reallocation
            self.notify_reallocation(True, ResourceType.BW)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Subprocess error in adding TC class: {e.stderr}", exc_info=True)
            self.logger.error(f"Command: {e.cmd}, Return code: {e.returncode}, Output: {e.output}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error in adding TC class: {e}", exc_info=True)


    def notify_reallocation(self, is_succeed: bool, resource_t: ResourceType, amount=None, clear_baseline=False) -> bool:
        try:
            self.logger.debug("Notifying reallocation for resource: %s", resource_t)
            resource_t_str = str(resource_t)
            resource_change = amount if amount is not None else (self.last_bw_change if resource_t == ResourceType.BW else self.last_cpu_change)
            notification = {
                "is_succeed": is_succeed,
                "actual_amount": resource_change,
                "resource_t": resource_t_str[resource_t_str.index(".") + 1:],
                "can_clear_profiling_flag": not self.last_bw_change_profiling,
                "is_min_bw": (self.min_bw) >= self.current_bw if not self.last_bw_change_profiling else False,
                "can_proceed": clear_baseline,
            }
            response: bool = self.stub.NotifyReallocation(pb2.Notification(**notification)).notified
            self.logger.debug("Reallocation notification response: %s", response)
            return response
        except Exception as e:
            self.logger.error(f"Error in notify_reallocation: {e}", exc_info=True)
            return False

    def start_profiling(self, resource_change: float, can_proceed: bool):
        try:
            self.logger.debug("Starting profiling with resource change: %f", resource_change)
            self.stub.StartProfile(pb2.ResourceChange(change=resource_change, can_proceed=can_proceed))
        except Exception as e:
            self.logger.error(f"Error in start_profiling: {e}", exc_info=True)

    def get_diff(self, resource_t_str) -> Tuple["PipelineRepr", InferDiff]:
        try:
            self.logger.debug("Getting resource difference for resource type: %s", resource_t_str)
            response = self.stub.GetDiff(pb2.ResourceType(resource_t=resource_t_str))
            diff = InferDiff(response.infer_diff, response.latency_diff, response.infer_diff_high, response.infer_diff_low,
                            response.bw_lower_bound, response.is_min_bw, response.curr_frame, response.curr_f1, response.curr_budget)
            self.logger.debug("Received resource difference: %s", diff)
            return self, diff
        except Exception as e:
            self.logger.error(f"Error in get_diff: {e}", exc_info=True)
            return self, None

    def cgroups_cpu(self, new_share: int, is_relative: bool) -> bool:
        try:
            self.logger.debug("Setting CPU share for container ID: %s", self.container_id)
            prev_cpu = self.current_cpu
            if is_relative:
                self.current_cpu += new_share
            else:
                self.current_cpu = new_share
            self.last_cpu_change = self.current_cpu - prev_cpu
            subprocess.run([*CG_SET, f"{CPU_SHR}={self.current_cpu}", self.container_id])
            return True
        except Exception as e:
            self.logger.error(f"Error in cgroups_cpu: {e}", exc_info=True)
            return False

    def tc(self, rate: float, is_relative: bool, is_profiling: bool) -> bool:
        try:
            self.logger.debug("Setting TC rate for container ID: %s", self.container_id)
            new_limit = self.current_bw + round(rate) if is_relative else round(rate)
            
            # Log before running the TC class replace command
            self.logger.debug("Running TC class replace command")
            subprocess.run([*TC_CLASS_REPLACE, IFB, "parent", "1:1",
                            "classid", f"1:{self.tc_class_id}",
                            "htb", "rate", f"{new_limit if not self.is_faked else new_limit - 250}kbit"], check=True)
            self.logger.info("TC class replace command successful")
            
            # Log before running the TC filter replace command
            self.logger.debug("Running TC filter replace command")
            try:
                subprocess.run([*TC_FILTER_REPLACE, IFB, "protocol", "ip", "parent", "1:", "prio", "1",
                                "u32", "match", "ip", "src", f"{self.client_uri}",
                                "match", "ip", "dport", f"{self.data_port}", "0xffff",
                                "flowid", f"1:{self.tc_class_id}"], check=True)
                self.logger.info("TC filter replace command successful")
            except subprocess.CalledProcessError as e:
                # If replace fails, attempt to add instead
                self.logger.warning(f"TC filter replace failed, attempting to add filter: {e}")
                subprocess.run([*TC_FILTER_ADD, IFB, "protocol", "ip", "parent", "1:", "prio", "1",
                                "u32", "match", "ip", "src", f"{self.client_uri}",
                                "match", "ip", "dport", f"{self.data_port}", "0xffff",
                                "flowid", f"1:{self.tc_class_id}"], check=True)
                self.logger.info("TC filter add command successful")

            self.last_bw_change = new_limit - self.current_bw if not self.last_bw_change_profiling else\
                new_limit - self.current_bw - self.last_bw_change

            if not is_profiling:
                self.current_bw = new_limit
                self.last_bw_change_profiling = False
            else:
                self.last_bw_change_profiling = True

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Subprocess error in setting TC rate: {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Error in setting TC rate: {e}", exc_info=True)
            return False

    def start_gpu_job(self):
        try:
            self.logger.debug("Starting GPU job for container ID: %s", self.container_id)
            self.stub.StartGPUJob(Empty())
        except Exception as e:
            self.logger.error(f"Error in start_gpu_job: {e}", exc_info=True)

    def end_inference(self):
        try:
            self.logger.debug("Ending inference for container ID: %s", self.container_id)
            self.stub.EndInference(Empty())
        except Exception as e:
            self.logger.error(f"Error in end_inference: {e}", exc_info=True)
