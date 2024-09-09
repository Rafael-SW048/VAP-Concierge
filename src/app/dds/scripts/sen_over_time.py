from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
import os
from typing import Dict, List, Optional, Tuple
import logging

import psutil

from app.dds.scripts.util import (
    Config,
    ProfileRow,
    Results,
    calculate_diff_concurrent,
    read_all_results,
    read_offline_result,
)

try: 
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 300
BATCH_SZ = 5

logging.basicConfig(filename='sense_over_time.log', level=logging.INFO)
processor: ProcessPoolExecutor = ProcessPoolExecutor(
    psutil.cpu_count(logical=False))


def sweep_bw_worker(infer: Results, gt: Results, profile_bw: List[ProfileRow]):
    return 1


def sweep_time_worker(
        profile_bw: List[ProfileRow], bw: int, max_bw: int,
        start_fid: int, all_res: Dict[str, Results], gt_dict: Results,
        process_pool: Optional[ProcessPoolExecutor] = None):

    filtered_profile_bw: List[ProfileRow] = [
            row for row in profile_bw
            if row.byte_sz <= bw and row.start_fid == start_fid]
    filtered_profile_max_bw: List[ProfileRow] = [
            row for row in profile_bw
            if row.byte_sz <= max_bw and row.start_fid == start_fid]
    end_fid = min(start_fid + BATCH_SZ, NFRAMES)

    if len(filtered_profile_bw) == 0:
        logging.error(
            f"Could not find profile_row with byte_sz less than {bw}")

    batch_gt = {
        fid: regions for fid, regions in gt_dict.items()
        if start_fid <= fid < end_fid
        }
    bw_f1: List[float] = []
    for row in filtered_profile_bw:
        config: Config = row.config
        vbatch_results_bw: Optional[Results] = None
        for profile_name, inference in all_res.items():
            if str(config) in profile_name:
                vbatch_results_bw = {
                    fid: regions for fid, regions in inference.items()
                    if start_fid <= fid < end_fid
                }
        if vbatch_results_bw is not None: 
            bw_f1.append(1 - calculate_diff_concurrent(
                vbatch_results_bw, batch_gt, 0.5, pool=process_pool).infer_diff)
        else:
            logging.debug("vbatch_results_bw is None")
    logging.debug(f"bw_f1: {bw_f1}")

    max_bw_f1: List[float] = []
    for row in filtered_profile_max_bw:
        config: Config = row.config
        vbatch_results_max_bw: Optional[Results] = None
        for profile_name, inference in all_res.items():
            if str(config) in profile_name:
                vbatch_results_max_bw = {
                    fid: region for fid, region in inference.items()
                    if start_fid <= fid < end_fid
                }
        if vbatch_results_max_bw is not None: 
            max_bw_f1.append(1 - calculate_diff_concurrent(
                vbatch_results_max_bw, batch_gt, 0.5,
                pool=process_pool).infer_diff)
        else:
            logging.debug("vbatch_results_max_bw is None")
    logging.debug(f"max_bw_f1: {max_bw_f1}")

    if max_bw_f1 is not None and bw_f1 is not None:
        logging.info(f"{start_fid}, {max(max_bw_f1) - max(bw_f1)}")
                

def get_sens(data_src: str, app_idx: int, step: int,
        bw: Optional[int]=None, time: Optional[int]=None):

    if bw is None and time is None:
        print("Must fixed either bandwidth or time!")
        exit(os.EX_USAGE)
    all_res: Dict[str, Results] = read_all_results(
        data_src, app_idx, 0, NFRAMES)
    gt_dict: Results = read_offline_result(
        data_src, app_idx, True, 0, NFRAMES)
    executor: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=psutil.cpu_count(logical=True))
    bw_profile: List[ProfileRow]
    with open(os.path.join(data_src, "profile_bw_frame"), "r")\
            as profile_bw_fd:
        bw_profile  = [ProfileRow(line) for line in profile_bw_fd]

    # if time is given, sweep the bandwidth and calculate sensitivity, and, if
    bw_profile: List[ProfileRow]
    # sensitivity is a function of res/bw and time.
    if time is not None:
        return
    if bw is not None:
        max_bw: int = bw + step
        [executor.submit(sweep_time_worker, bw_profile, bw, max_bw, start_fid,
                         all_res, gt_dict, processor)
         for start_fid in range(0, NFRAMES, BATCH_SZ)
        ]
    executor.shutdown()


def main():
    get_sens("/home/cc/tmpfs/trafficcam_1/merged_results", 1, 30000, 50000)
    # get_sens("/home/cc/tmpfs/trafficcam_2/merged_results", 2, 30000, 50000)


if __name__ == "__main__":
    main()
