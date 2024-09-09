from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing
from multiprocessing.connection import _ConnectionBase
import os
from threading import Event, Lock
from time import perf_counter_ns as perf_counter
from typing import Any

import grpc
from google.protobuf.empty_pb2 import Empty

from api.common.enums import ResourceType
from api.common.typedef import Callback
from api.pipeline_repr import InferDiff
import api.proto.app_server_pb2 as app_pb2
import api.proto.app_server_pb2_grpc as app_pb2_grpc
import api.proto.concierge_pb2 as concierge_pb2
import api.proto.concierge_pb2_grpc as concierge_pb2_grpc
from datetime import datetime

from dds_utils import writeResult


HIST_WEIGHT: float = float(os.getenv("HIST_WEIGHT") or "0.3")
CONCIERGE_URI: str = os.getenv("CONCIERGE_URI") or "127.0.0.1:5000"
HOSTNAME: str = os.getenv("HOSTNAME") or "App"


class AppServer(app_pb2_grpc.PipelineServicer, ABC):

    def __init__(self, uri: str, edge_uri: str, control_port: int, data_port: int, is_faked: bool) -> None:
        self.parent_conn: _ConnectionBase
        self.child_conn: _ConnectionBase
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.concierge_uri: str = CONCIERGE_URI
        self.container_id: str = HOSTNAME
        self.is_faked = is_faked

        # setup handlers for loggers
        self._logger = logging.getLogger(f"AppServer.{self.container_id}")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler("./app_server.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-6s %(name)-14s %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.INFO)

        # connect with concierge
        self.channel = grpc.insecure_channel(
                f"{self.concierge_uri}")
        self.concierge_stub = concierge_pb2_grpc.ConciergeStub(
                self.channel)

        # set options according to the args
        self.data_port = data_port
        self.control_port = control_port
        self.uri: str = f"{uri}:{self.control_port}"
        self.client_uri = edge_uri
        self.app_info: concierge_pb2.AppInfo = concierge_pb2.AppInfo(
            container_id=self.container_id,
            server_uri=self.uri,
            client_uri=uri,
            server_port=self.data_port,
            is_faked=self.is_faked,
            pid=128947)
        self.callback: Callback = self.default_callback

        self.is_profiling = Event()

        # bw related fields
        self.current_bw: int = 0
        self.is_min_bw: bool = False

        # gpu related fields
        self.token_lock: Lock = Lock()
        self.token_bucket: int = 0
        self.estimated_infer_time: int = 0
        self.infer_time_lock: Lock = Lock()
        self.can_proceed_gpu_task: Event = Event()
        self.should_end_gpu_task: Event = Event()

    # grpc service
    def GetDiff(self, request, _) -> app_pb2.Diff:
        # while self.start_frame < self.reference:
        #     sleep(0.5)
        # Turunin monitor intervalnya
        self.is_profiling.set()
        startRealloc = datetime.now()
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])
        self._logger.info("Start Profiling")
        # di sini harusnya masih bareng

        resource_t_str: str = request.resource_t
        self.prep_profiling()
        self.concierge_stub.ReadyToProfile(
            concierge_pb2.PID(container_id=self.container_id,
                              pid=os.getpid(),
                              resource_t=resource_t_str))
        # block until resource manager change the resource limit and notify the
        # app can start profiling
        try:
            self.resource_change = int(self.child_conn.recv())

            # start profiling timer
            st = perf_counter()
            res = self.get_diff()
            # writeResult(self.uri, res, "messageDebug")
            self._logger.debug(f"get_diff(): {perf_counter() - st}")
            # METHOD 3
            return app_pb2.Diff(
                    infer_diff = res.infer_diff,
                    latency_diff = res.latency_diff,
                    infer_diff_high = res.infer_diff_high,
                    infer_diff_low = res.infer_diff_low,
                    bw_lower_bound = res.bw_lower_bound,
                    is_min_bw = res.is_min_bw,
                    curr_frame = res.curr_frame,
                    curr_f1 = res.curr_f1,
                    curr_budget = res.curr_budget)
        except ValueError:
            return app_pb2.Diff(
                    infer_diff=-1,
                    latency_diff=-1,
                    infer_diff_high=-1,
                    infer_diff_low=-1,
                    bw_lower_bound=-1,
                    is_min_bw = self.is_min_bw)

    def NotifyReallocation(self, request, _) -> app_pb2.IsNotified:
        resource_t: ResourceType = ResourceType[request.resource_t]
        if request.can_clear_profiling_flag:
            self.is_profiling.clear()
            self.is_min_bw = request.is_min_bw
        st = perf_counter()
        is_succeed = self.callback(
            request.is_succeed, request.actual_amount, resource_t)
        if request.can_proceed:
            self.is_allocated = True
        self._logger.debug(f"callback(): {perf_counter() - st}")
        return app_pb2.IsNotified(notified=is_succeed)

    def StartProfile(self, request, _) -> app_pb2.IsNotified:
        if request.can_proceed:
            self.parent_conn.send(request.change)
        else:
            self._logger.error("Stop profiling")
            self.parent_conn.send("stop")
        return app_pb2.IsNotified(notified=True)

    def RefillToken(self, request, _) -> app_pb2.IsNotified:
        refill_amount: int = request.amount
        self.token_lock.acquire()
        self.token_bucket += refill_amount
        self.token_lock.release()
        return app_pb2.IsNotified(notified=True)

    def StartGPUJob(self, _, __):
        self.can_proceed_gpu_task.set()
        return app_pb2.IsNotified(notified=True)
    
    def EndInference(self, _, __):
        self.should_end_gpu_task.set()
        self._logger.warning("End Inference received, set.")
        return app_pb2.IsNotified(notified=True)
    
    def done_inference(self) -> bool:
        status = self.concierge_stub.DoneInference(self.app_info)
        self._logger.warning(f"gRPC Done Inference notified: {status.is_succeed}")
        return status.is_succeed
    
    # experimental
    def get_best_bw_baseline(self, f1_scores) -> bool:
        baseline_info = concierge_pb2.BaselineInfo(
            container_id=self.container_id,
            f1_scores=f1_scores)
        status = self.concierge_stub.BaselineBW(baseline_info)
        return status.is_succeed


    # public methods
    def checkin(self) -> bool:
        status = self.concierge_stub.CheckinApp(self.app_info)
        return status.is_succeed

    @staticmethod
    def _infer_wrapper(infer_func):

        def intercept_infer_func(self: AppServer, *args, **kargs):
            # check current token bucket has enough toekn.
            # if not wait for another refill
            has_enough_token: bool =\
                self.token_bucket >= self.estimated_infer_time
            while not has_enough_token:
                has_enough_token: bool =\
                    self.token_bucket >= self.estimated_infer_time

            # wait for scheduler's signal to start gpu task
            # mainly waiting for other jobs on the gpu to finish
            self.concierge_stub.SubmitGPUJob(self.app_info)
            while not self.can_proceed_gpu_task.is_set():
                pass
            # run inference and record elapsed time
            st: int = perf_counter()
            ret = infer_func(self, *args, **kargs)
            elapsed: int = perf_counter() - st

            # notify gpu scheduler the gpu task is done
            self.concierge_stub.DoneGPUJob(Empty())

            # update the ewma of the inference time and the token_bucket
            self.token_lock.acquire()
            self.token_bucket -= elapsed
            self._logger.info(
                f"Used {elapsed} tokens. Has {self.token_bucket} token left.")
            self.token_lock.release()

            self.estimated_infer_time = int(
                self.estimated_infer_time * HIST_WEIGHT
                + elapsed * (1 - HIST_WEIGHT))
            return ret

        return intercept_infer_func

    # abstract methods
    @abstractmethod
    def default_callback(self, is_succeed: bool, resource_change: float,
                         resource_t: ResourceType) -> bool:
        self._logger.error("Abstract Method default_callback not Implemented")
        raise NotImplementedError

    @abstractmethod
    def prep_profiling(self) -> Any:
        self._logger.error("Abstract Method adapt not Implemented")
        raise NotImplementedError

    @abstractmethod
    def get_diff(self) -> InferDiff:
        self._logger.error("Abstract Method get_diff not Implemented")
        raise NotImplementedError


def serve(servicer: AppServer):
    server = grpc.server(ThreadPoolExecutor(4))
    app_pb2_grpc.add_PipelineServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{servicer.control_port}")
    server.start()
    server.wait_for_termination()
