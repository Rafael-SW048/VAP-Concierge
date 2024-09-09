import argparse
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
import os
from threading import Lock, Thread
import time
from typing import Dict, List, Tuple, Any
import logging 

from api.app_server import AppServer
from api.app_server import serve as grpc_serve
from api.common.enums import ResourceType
from api.pipeline_repr import InferDiff
from app.dds.scripts.util import (
    Config,
    ProfileRow,
    Region,
    calculate_diff_concurrent,
    copy_inference_dict,
    inference_to_str_lst,
    read_all_results,
    read_offline_result,
    read_cache_result,
    read_all_profile,
    get_byte_sz,
    get_gt_byte_sz
)

Box = List[float]

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


def _callback_worker(profile_row: ProfileRow,
                    inference_config: Dict[int, List[Region]],
                    inference_gt: Dict[int, List[Region]],
                    iou_threshold) -> Tuple[Config, float, int]:
    f1 = 1 - calculate_diff_concurrent(
            inference_config, inference_gt, iou_threshold,
            pool=process_pool).infer_diff
    return profile_row.config, f1, profile_row.byte_sz


class DDS(AppServer):

    def __init__(self, client_uri: str,
                 control_port: int, out_file_path: str, app_idx: int) -> None:
        super().__init__(client_uri, client_uri, control_port)
        self.current_bw: int
        self.config: Config = Config(0.1, 0.4, 38, 28)
        self.curr_fid: int = 0
        self.iou_threshold: float = 0.5
        self.app_idx: int = app_idx
        self.out_file_path: str = out_file_path
        self.data_src: str =\
            f"/home/cc/tmpfs/dds{self.app_idx}/workspace/merged_results/"
        self.gt_dict: Dict[int, List[Region]] = read_offline_result(
            self.data_src, app_idx, True, 0, NFRAMES)
        self.all_results: Dict[str, Dict[int, List[Region]]] =\
            read_all_results(self.data_src, self.app_idx, 0, NFRAMES,
                             process_pool)
        self.prev_inference: Dict[int, List[Region]]
        self.prev_cache_lock: Lock = Lock()
        self.will_backlog: bool = False
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor()

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

    def get_diff(self) -> InferDiff:

        if self.will_backlog:
            self.logger.error("Will backlock")
            return InferDiff(1, 1)

        self.prev_cache_lock.acquire()
        self.logger.debug("get_diff() acquired prev_cache_lock")
        # get the start_fid for the cached inference results
        try:
            start_fid = min(self.prev_inference.keys())
            end_fid = min(start_fid + BATCH_SZ, NFRAMES)
        except AttributeError:
            self.logger.error(
                "No Previous Inference. Cannot get profiling start_fid .")
            self.prev_cache_lock.release()
            self.logger.debug("get_diff() released prev_cache_lock")
            return InferDiff(-1, -1)

        profiling_inference = read_cache_result(
                self.all_results, self.app_idx, start_fid, end_fid,
                *(self.config.pack()))
        byte_sz: int  = get_byte_sz(self.data_src, self.config, start_fid)

        # CHEAT_MODE directly calculate the sensitivity instead of using
        # inference difference for approximation as in non-CHEAT_MODE.
        if DEBUG:
            self.logger.debug("Enter get_diff() DEBUG section")
            max_bw_down: int = self.current_bw - PROFILING_DELTA * 2
            self.logger.debug(f"Up: {self.current_bw} Down: {max_bw_down}")
            gt_dict: Dict[int, List[Region]] = {
                fid: self.gt_dict[fid]
                for fid in range (start_fid, end_fid)
                }
            all_profile: List[ProfileRow] = read_all_profile(
                self.data_src, start_fid)
            if len(all_profile) == 0:
                self.logger.error(
                    f"Cannot find ProfileRow with start_fid {start_fid}")
                self.prev_cache_lock.release()
                self.logger.debug("get_diff() released prev_cache_lock")
                self.logger.debug(f"Profiling byte: {byte_sz}")
                return InferDiff(-1, -1)
            self.logger.debug("DEBUG read_all_profile() done")

            cache_f1: float = 1 - calculate_diff_concurrent(
                self.prev_inference, gt_dict, self.iou_threshold,
                pool=process_pool).infer_diff

            # get acc_up
            adapt_res_up = self._adapt(
                start_fid, end_fid, self.current_bw, all_profile)
            max_f1_up = adapt_res_up[2] if not adapt_res_up[0] else 1
            acc_diff_up: float = InferDiff(
                max_f1_up - cache_f1, 0).infer_diff
            self.logger.debug("DEBUG _adapt() for up done")
            byte_sz += get_gt_byte_sz(self.data_src, start_fid)

            # get acc_down
            adapt_res_down = self._adapt(
                start_fid, end_fid, max_bw_down, all_profile)
            max_f1_down = adapt_res_down[2] if not adapt_res_down[0] else 1
            self.logger.debug("DEBUG _adapt() for down done")
            acc_diff_down: float = max_f1_down - cache_f1

            profiling_inference = read_cache_result(
                    self.all_results, self.app_idx, start_fid, end_fid,
                    *(adapt_res_up[1].pack()))
            infer_diff_up: float = calculate_diff_concurrent(
                self.prev_inference, profiling_inference, self.iou_threshold,
                pool=process_pool).infer_diff
            self.logger.debug(
                f"Diff{self.app_idx}: {acc_diff_up} {acc_diff_down}"
                f" {infer_diff_up} {start_fid}")

            # if in CHEAT_MODE, we use accuracy difference instead of inference
            # difference
            if CHEAT_MODE:
                diff: InferDiff = InferDiff(acc_diff_up, 0)
            else: 
                diff: InferDiff = InferDiff(infer_diff_up, 0)
        else: 
            diff: InferDiff = calculate_diff_concurrent(
                self.prev_inference, profiling_inference, self.iou_threshold,
                pool=process_pool)

        self.prev_cache_lock.release()
        self.logger.debug("get_diff() released prev_cache_lock")
        self.logger.debug(f"Profiling byte: {byte_sz}")

        # discard this profiling iteration if the diff/sens is out of boundary
        if 0 <= diff.infer_diff < MAX_DIFF_THRESHOLD:
            if diff.infer_diff <= MIN_DIFF_THRESHOLD:
                return InferDiff(0, 0)
            return diff
        else: 
            self.logger.error(f"Discarded {diff.infer_diff}")
            return InferDiff(-1, -1)

    def default_callback(self, _: bool, resource_change: float,
                         resource_t: ResourceType) -> bool:
        if resource_t == ResourceType.BW:
            self.current_bw += int(resource_change)
            self.logger.info(
                f"Received bandwidth allocation: {self.current_bw}")
            try:
                start_fid = min(self.prev_inference.keys())
                end_fid = min(start_fid + BATCH_SZ, NFRAMES)
            except AttributeError:
                start_fid = min(self.curr_fid,
                                NFRAMES)
                end_fid = min(NFRAMES, start_fid + BATCH_SZ)
            all_profile: List[ProfileRow] = read_all_profile(
                self.data_src, start_fid)
            # no profile row whose fid equals current-fid found
            if len(all_profile) == 0:
                return False

            # find the config produce the highest f1 score
            max_f1: float
            max_config: Config
            max_config_bw: int
            self.will_backlog, max_config, max_f1, max_config_bw =\
                self._adapt(
                    start_fid, end_fid, self.current_bw, all_profile)
            self.config = max_config
            if not self.will_backlog:
                self.logger.info(
                    f"Callback changes the configuration to"
                    f" {self.config} f1 {max_f1} byte_sz {max_config_bw}")
            else:
                self.logger.info(
                    f"Callback changes the configuration to"
                    f" {self.config} f1 {max_f1} byte_sz {max_config_bw}"
                    "due to backlog")
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

    def _adapt(self, start_fid: int, end_fid: int, max_bw: int,
               all_profile: List[ProfileRow])\
            -> Tuple[bool, Config, float, int]:
        all_profile.sort(key=lambda profile_row: profile_row.byte_sz)
        backlog: bool = False
        inference_gt: Dict[int, List[Region]] = {
            fid: self.gt_dict[fid] for fid in range(start_fid, end_fid)}
        # if no config can satisfy current bandwidth limit,
        # return the config requires the least amount of bandwidth
        if all_profile[0].byte_sz > max_bw:
            self.logger.error(
                "Cannot find satisfied configuration under current limit")
            backlog = True
            config = all_profile[0].config
            inference_config = read_cache_result(
                self.all_results, self.app_idx, start_fid, end_fid,
                config.low_res, config.high_res,
                config.low_qp, config.high_qp)
            return (backlog, *_callback_worker(
                all_profile[0], inference_config, inference_gt,
                self.iou_threshold))
        # if there is at least one config satisfying current bandwidth limit,
        # calculate the f1 score for each config
        else:
            futures_f1: List[Future] = []
            for profile_row in all_profile:
                if (profile_row.byte_sz <= max_bw):
                    config = profile_row.config
                    inference_config = read_cache_result(
                        self.all_results, self.app_idx, start_fid, end_fid,
                        config.low_res, config.high_res,
                        config.low_qp, config.high_qp)
                    futures_f1.append(self.executor.submit(
                        _callback_worker,
                        profile_row, inference_config, inference_gt,
                        self.iou_threshold
                        ))
            try:
                # find the config produce the highest f1 score
                results = [future.result() for future in futures_f1] 
                return (backlog, *max(results, key=lambda result: result[1]))
            except Exception as e:
                self.logger.critical(e)
                exit(-1)


    def run(self):
        while self.current_bw is None or self.config is None:
            pass
        self.curr_fid = 0
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
            # find the best config for this batch
            with open(os.path.join(self.data_src, "profile_bw_frame"), "r")\
                    as profile_bw_fd:
                all_profile: List[ProfileRow] = [
                    ProfileRow(line) for line in profile_bw_fd
                    if ProfileRow(line).start_fid == start_fid
                    ]
            self.will_backlog, config, max_f1, max_config_bw = self._adapt(
                start_fid, end_fid, self.current_bw, all_profile)
            self.config = config
            self.logger.info(
                f"Main loop changes the configuration to {self.config}"
                f" f1 {max_f1} byte_sz {max_config_bw}")

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

            self.prev_cache_lock.acquire()
            self.logger.debug("Main loop acquired prev_cache_lock")
            self.prev_inference = copy_inference_dict(detections)
            self.prev_cache_lock.release()
            self.logger.debug("Main loop released prev_cache_lock")
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
        OUT_AFFIX  = ""
    if BASELINE_MODE:
        OUT_AFFIX = OUT_AFFIX + "_baseline"
    print(OUT_AFFIX)
    print(CHEAT_MODE)
    dds = DDS("localhost", 10000 + args.app_idx,
              f"./sim/dds{args.app_idx}{OUT_AFFIX}", args.app_idx)
    Thread(target=grpc_serve, args=(dds,)).start()
    time.sleep(5)
    dds.checkin()
    st = time.perf_counter_ns()
    dds.run()
    print(f"dds: {time.perf_counter_ns() - st}")
