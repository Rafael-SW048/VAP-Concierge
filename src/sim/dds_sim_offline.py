import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from threading import Thread
import time
from typing import Any, Dict, List, Optional, Tuple

from api.app_server import AppServer
from api.app_server import serve as grpc_serve
from api.common.enums import ResourceType
from api.pipeline_repr import InferDiff
from app.dds.scripts.util import (
    Config,
    ProfileRow,
    Region,
    get_byte_sz,
    inference_to_str_lst,
    read_all_profile,
    read_all_results,
    read_cache_result,
    read_offline_result,
)

Box = List[float]

# read experiment setting from envrionment variables
BATCH_SZ = 5
SEARCH_RNG = 20000
NPROCESS = BATCH_SZ
try:
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 100
try:
    BASELINE_MODE = eval(os.environ["BASELINE_MODE"])
except KeyError or ValueError:
    BASELINE_MODE = False
try:
    CHEAT_MODE = eval(os.environ["CHEAT_MODE"])
except KeyError or ValueError:
    CHEAT_MODE = False
try:
    DEBUG = eval(os.environ["DEBUG"])
except KeyError or ValueError:
    DEBUG = False
try:
    MAX_DIFF_THRESHOLD = float(os.environ["MAX_DIFF_THRESHOLD"])
except KeyError or ValueError:
    MAX_DIFF_THRESHOLD = 0.75
try:
    MIN_DIFF_THRESHOLD = float(os.environ["MIN_DIFF_THRESHOLD"])
except KeyError or ValueError:
    MIN_DIFF_THRESHOLD = 0.1
try:
    PROFILING_DELTA = int(os.environ["PROFILING_DELTA"])
except KeyError:
    PROFILING_DELTA = 5000

process_pool: ProcessPoolExecutor = ProcessPoolExecutor()


@dataclass
class SamplePoint:
    bw_bin: int
    bw_used: int
    f1: float
    config: Config


class DDS(AppServer):

    def __init__(self, client_uri: str, need_concierge: bool,
                 control_port: int, out_file_path: str, app_idx: int) -> None:
        super().__init__(client_uri, client_uri, control_port)
        self.current_bw: int
        self.config: Config = Config(0.1, 0.4, 38, 28)
        self.curr_fid: int = 0
        self.iou_threshold: float = 0.5
        self.app_idx: int = app_idx
        self.out_file_path: str = out_file_path
        self.will_backlog: bool = False
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor()

        # setup logger
        self.logger = logging.getLogger(self.container_id)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(f"./dds{self.app_idx}.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-6s %(name)-14s %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        # read data from data_src
        self.data_src: str =\
            f"/home/cc/tmpfs/dds{self.app_idx}/workspace/merged_results/"
        self.gt_dict: Dict[int, List[Region]] = read_offline_result(
            self.data_src, app_idx, True, 0, NFRAMES)
        self.all_results: Dict[str, Dict[int, List[Region]]] =\
            read_all_results(self.data_src, self.app_idx, 0, NFRAMES,
                             process_pool)
        self.profile_curve: List[SamplePoint] = self._read_bw_profile()
        self.prev_smpl_p: Optional[SamplePoint] = None
        self.curr_smpl_p: SamplePoint = self.profile_curve[0]

    def get_diff(self) -> InferDiff:
        if self.prev_smpl_p is not None:
            return InferDiff(self.curr_smpl_p.f1 - self.prev_smpl_p.f1, 0)
        else:
            return InferDiff(-1, 0)

    def default_callback(self, _: bool, resource_change: float,
                         resource_t: ResourceType) -> bool:
        if resource_t == ResourceType.BW:
            self.current_bw += int(resource_change)
            self.logger.info(
                f"Received bandwidth allocation: {self.current_bw}")
            _, new_smpl_point = self._adapt()
            self.prev_smpl_p = self.curr_smpl_p
            self.curr_smpl_p = new_smpl_point
            self.config = new_smpl_point.config
            self.logger.info(
                f"Change to {self.config} with f1 {new_smpl_point.f1}")
            return True
        return False

    def prep_profiling(self) -> Any:
        pass

    def _read_next_batch_bw(self, config: Config) -> int:
        all_profile: List[ProfileRow] = read_all_profile(
            self.data_src, self.curr_fid)
        for profile_row in all_profile:
            if profile_row.config.is_equal(config):
                return profile_row.byte_sz
        return -1

    def _read_bw_profile(self) -> List[SamplePoint]:
        profile_curve: List[SamplePoint] = []
        with open(os.path.join(self.data_src, "pf"), "r") as bw_profile_fd:
            for line in bw_profile_fd:
                line_lst: List[str] = line.split(" ")
                try:
                    config_str: str = line_lst[3]
                    config_str_lst: List[str] = config_str.split("_")
                    smpl_p: SamplePoint = SamplePoint(
                        bw_bin=int(float(line_lst[0])),
                        bw_used=int(line_lst[1]),
                        f1=float(line_lst[2]),
                        config=Config(
                            low_res=float(config_str_lst[0]),
                            high_res=float(config_str_lst[1]),
                            low_qp=int(config_str_lst[2]),
                            high_qp=int(config_str_lst[3])
                            )
                        )
                    profile_curve.append(smpl_p)
                except ValueError:
                    self.logger.error("Cannot parse current pf line.")
                    pass
        return profile_curve

    def _adapt(self) -> Tuple[bool, SamplePoint]:
        max_f1_smpl_p: SamplePoint = self.profile_curve[0]
        for smpl_p in self.profile_curve:
            if (smpl_p.f1 > max_f1_smpl_p.f1
                    and smpl_p.bw_used <= self.current_bw):
                max_f1_smpl_p = smpl_p
        will_backlog: bool =\
            self._read_next_batch_bw(max_f1_smpl_p.config) > self.current_bw
        return will_backlog, max_f1_smpl_p

    def run(self):
        while self.current_bw is None or self.config is None:
            pass
        self.curr_fid = 50
        out_fd = open(self.out_file_path, "w+")
        total_byte: int = 0
        total_latency: float = 0
        while self.curr_fid < NFRAMES:
            while self.is_profiling.is_set():
                time.sleep(1)
            start_fid = self.curr_fid
            end_fid = min(start_fid + BATCH_SZ, NFRAMES)
            self.logger.info(
                f"Running inference from {start_fid} to {end_fid}")
            while self.is_profiling.is_set():
                time.sleep(1)

            # fake thinking time and then write inference result to output file
            time.sleep(5)
            if DEBUG:
                real_next_batch_bw: int = self._read_next_batch_bw(self.config)
                self.will_backlog: bool = real_next_batch_bw > self.current_bw
                total_latency += real_next_batch_bw / self.current_bw
                self.logger.debug(
                    f"will_backlog: {self.will_backlog}"
                    f" real: {real_next_batch_bw} alloc: {self.current_bw}")
            detections = read_cache_result(
                self.all_results, self.app_idx, start_fid, end_fid,
                *(self.config.pack()))
            total_byte += get_byte_sz(self.data_src, self.config, start_fid)
            inference_str_lst = inference_to_str_lst(detections)
            for region in inference_str_lst:
                out_fd.write(f"{region}" + "\n")

            while self.is_profiling.is_set():
                time.sleep(1)

            self.curr_fid += BATCH_SZ

        out_fd.close()
        self.logger.info("Done")
        self.logger.debug(f"Total Byte Size: {total_byte}")
        self.logger.debug("Average transimission latency:"
                          f" {total_latency / (NFRAMES / BATCH_SZ)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--app_idx",
                        help="App ID",
                        type=int,
                        default=0)
    parser.add_argument("-c", "--client_ip",
                        help="Edge client ip address without port",
                        type=str,
                        required=True)
    args = parser.parse_args()
    try:
        OUT_AFFIX = os.environ["OUT_AFFIX"]
    except KeyError or ValueError:
        OUT_AFFIX = ""
    if BASELINE_MODE:
        OUT_AFFIX = OUT_AFFIX + "_baseline"
    print(OUT_AFFIX)
    print(CHEAT_MODE)
    dds = DDS("localhost", True, 10000 + args.app_idx,
              f"./sim/dds{args.app_idx}{OUT_AFFIX}", args.app_idx)
    Thread(target=grpc_serve, args=(dds,)).start()
    time.sleep(5)
    dds.checkin()
    st = time.perf_counter_ns()
    dds.run()
    print(f"dds: {time.perf_counter_ns() - st}")
