from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import functools
import os
from typing import Dict, List, Tuple
from operator import add
import numpy as np

from app.dds.scripts.util import (
    ProfileRow,
    calculate_diff,
    calculate_diff_concurrent,
    read_all_results,
    read_offline_result,
    Results,
    AllResults
)

try: 
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 300
BATCH_SZ = 5


process_pool: ProcessPoolExecutor = ProcessPoolExecutor()


def get_profile_total_bw(profile_name: str, data_src: str,
                         end_fid: int, video_length: int):
    total_bw: int = 0
    with open(os.path.join(data_src, "profile_bw_frame"), "r")\
            as profile_bw_frame:
        for line in profile_bw_frame:
            profile_row = ProfileRow(line)
            if (str(profile_row.config) in profile_name
                    and profile_row.end_fid <= end_fid):
                total_bw += profile_row.byte_sz
    return int(total_bw / video_length)


def worker(profile_name: str, data_src: str, end_fid: int,
           video_length: int, gt_dict: Results,
           profile_res: Results)\
    -> Tuple[str, int, float]:
    return (profile_name,
            get_profile_total_bw(
                profile_name, data_src, end_fid, video_length),
            1 - calculate_diff_concurrent(profile_res, gt_dict, 0.5,
                                          pool=process_pool)
            .infer_diff)


def gen_pf(end_fid: int, start_bw: int, end_bw: int, bin_sz: int,
           video_length: int, data_src: str,
           gt_dict: Results,
           all_res: AllResults,
           is_total: bool,
           std_out: bool = False) -> List[Tuple[str, int, float]]:

    if is_total:
        with ThreadPoolExecutor() as executor:
            results: List[Future] = [
                    executor.submit(worker, profile_name, data_src, end_fid,
                                    video_length, gt_dict,
                                    all_res[profile_name]) 
                    for profile_name in all_res.keys()
                    ]
            all_bw_f1: List[Tuple[str, int, float]] = [
                    result.result() 
                    for result in results
                    ]
                
    else:
        all_bw_f1: List[Tuple[str, int, float]] = []
        for profile_name, inference in all_res.items():
            batch_inference: Results = {
                fid: inference[fid]
                for fid in range(end_fid - BATCH_SZ, end_fid)
            }
            batch_gt: Results = {
                fid: gt_dict[fid]
                for fid in range(end_fid - BATCH_SZ, end_fid)
            }
            batch_f1: float =\
                1 - calculate_diff(batch_inference, batch_gt, 0.5)\
                .infer_diff
            batch_byte_sz: int = -1
            with open(os.path.join(data_src, "profile_bw_frame"), "r")\
                    as profile_bw_frame:
                for line in profile_bw_frame:
                    profile_row = ProfileRow(line)
                    if (str(profile_row.config) in profile_name
                            and profile_row.end_fid == end_fid):
                        batch_byte_sz = profile_row.byte_sz
            all_bw_f1.append((profile_name, batch_byte_sz, batch_f1))

    pf: List[Tuple[str, int, float]] = []
    for bin_start in range(start_bw, end_bw, bin_sz):
        bin_end = bin_start + bin_sz
        bin_max: Tuple[str, int, float] = ("", -1, -1)
        for res in all_bw_f1:
            if res[1] <= bin_end and res[2] > bin_max[2]:
                bin_max = res
        start: int = bin_max[0].index("dds_") + 4
        end: int = bin_max[0].index("_0.0")
        config_str_short: str = bin_max[0][start: end]
        if std_out:
            try:
                start: int = bin_max[0].index("dds_") + 4
                end: int = bin_max[0].index("_0.0")
                config_str_short: str = bin_max[0][start: end]
                print((bin_start + bin_end) / 2, bin_max[1], bin_max[2],
                      config_str_short)
            except ValueError:
                pass
        pf.append((config_str_short, bin_max[1], bin_max[2]))
    return pf


def gen_avg_pf(end_fid: int, start_bw: int, end_bw: int, bin_sz: int,
           video_length: int, data_src_all: Dict[str, str],
           gt_dict_all: Dict[str, Results], all_res_all: Dict[str, AllResults],
           is_total: bool, std_out: bool = False):
    pf_all: Dict[str, List[Tuple[int, float]]] = {}
    for video_name in data_src_all.keys():
        data_src: str = data_src_all[video_name]
        gt_dict: Results = gt_dict_all[video_name]
        all_res: AllResults = all_res_all[video_name]
        with ThreadPoolExecutor() as executor:
            results: List[Future] = [
                    executor.submit(worker, profile_name, data_src, end_fid,
                                    video_length, gt_dict,
                                    all_res[profile_name]) 
                    for profile_name in all_res.keys()
                    ]
            all_bw_f1: List[Tuple[str, int, float]] = [
                    result.result() 
                    for result in results
                    ]
        for point in all_bw_f1:
            start: int = point[0].index("dds_") + 4
            end: int = point[0].index("_0.0")
            config_name: str = point[0][start: end]
            if config_name not in pf_all.keys():
                pf_all[config_name] = [(point[1], point[2])]
            else:
                pf_all[config_name].append((point[1], point[2]))

    pf_aggregate: List[Tuple[str, float, float]] = [(
        config_name,
        *(np.array(functools.reduce(
            lambda prev, curr: (prev[0] + curr[0], prev[1] + curr[1]),
            pf_all[config_name])
        ) / len(pf_all[config_name])))
        for config_name in pf_all
        ]

    pf_final: List[Tuple[str, float, float]] = []
    for bin_start in range(start_bw, end_bw, bin_sz):
        bin_end = bin_start + bin_sz
        bin_max: Tuple[str, float, float] = ("", -1, -1)
        for res in pf_aggregate:
            if res[1] <= bin_end and res[2] > bin_max[2]:
                bin_max = res
        if std_out:
            print(
                (bin_end + bin_start) / 2, bin_max[1], bin_max[2], bin_max[0])
        pf_final.append(bin_max)


def main():
    try:
        all_videos: bool = eval(os.environ["ALL_VID"])
    except ValueError or KeyError:
        all_videos: bool = False
    if all_videos: 
        data_src_all = {
                "rene": "/home/cc/data-set/rene/merged_results",
                "uav-1": "/home/cc/data-set/uav-1/merged_results",
                "uav-2": "/home/cc/data-set/uav-2/merged_results",
                "thailand-3": "/home/cc/data-set/thailand-3/merged_results",
                }
        gt_dict_all: Dict[str, Results] = {}
        all_res_all: Dict[str, AllResults] = {}
        for video_name, data_src in data_src_all.items():
            gt_dict_all[video_name] = read_offline_result(
                data_src, video_name, True, 0, NFRAMES)
            all_res_all[video_name] = read_all_results(
                data_src, video_name, 0, NFRAMES)

        gen_avg_pf(end_fid=NFRAMES, start_bw=20000, end_bw=300000, bin_sz=4000,
               video_length=20, data_src_all=data_src_all,
               gt_dict_all=gt_dict_all,
               all_res_all=all_res_all,
               is_total=True,
               std_out=True)

    else:

        try:
            dds_id: int = int(os.environ["DDS_ID"])
            data_src: str = os.environ["MERGED"]
            video_length: int = int(os.environ["VID_LEN"])
        except KeyError or ValueError:
            return -1
        gt_dict: Results = read_offline_result(
            data_src, dds_id, True, 0, NFRAMES)
        all_res: AllResults = read_all_results(
            data_src, dds_id, 0, NFRAMES)
        gen_pf(end_fid=NFRAMES, start_bw=20000, end_bw=300000, bin_sz=4000,
               video_length=video_length, data_src=data_src,
               gt_dict=gt_dict,
               all_res=all_res,
               is_total=True,
               std_out=True)


if __name__ == "__main__":
    main()
