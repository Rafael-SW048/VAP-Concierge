from dataclasses import dataclass
import os
from typing import Dict, List, Union, Tuple, Optional

from app.dds.scripts.pareto_front import gen_pf
from app.dds.scripts.util import (
    ProfileRow,
    Region,
    calculate_diff_concurrent,
    read_all_results,
    read_offline_result,
)

try: 
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 300
BATCH_SZ = 5


@dataclass
class BatchRes:
    profile_row: ProfileRow
    inference: Dict[int, List[Region]]
    f1: float


@dataclass
class Diff:
    inference_diff: float
    f1_diff: float


def get_profile_row(profile_name: str, data_src: str,
                    start_fid: int, end_fid: int) -> Union[ProfileRow, None]:
    with open(os.path.join(data_src, "profile_bw_frame"), "r")\
            as profile_bw_frame:
        for line in profile_bw_frame:
            profile_row = ProfileRow(line)
            if (str(profile_row.config) in profile_name
                    and profile_row.end_fid == end_fid
                    and profile_row.start_fid == start_fid):
                return profile_row
    return None


def parse_batch_res(data_src: str, start_fid: int, end_fid: int,
                    gt_dict: Dict[int, List[Region]],
                    all_res: Dict[str, Dict[int, List[Region]]])\
        -> Dict[int, List[BatchRes]]:
    res: Dict[int, List[BatchRes]] = {
            batch_first_fid: [] for batch_first_fid
            in range(start_fid, end_fid, BATCH_SZ)
            }
    for profile_name, inference in all_res.items():
        for start in range(start_fid, end_fid, BATCH_SZ):
            end = start + BATCH_SZ
            batch_inference: Dict[int, List[Region]] = {
                    fid: inference[fid] for fid in range(start, end)
                    }
            gt: Dict[int, List[Region]] = {
                    fid: gt_dict[fid] for fid in range(start, end)
                    }
            batch_f1: float =\
                1 - calculate_diff_concurrent(
                    batch_inference, gt, 0.5).infer_diff
            profile_row = get_profile_row(profile_name, data_src, start, end)
            if profile_row is not None:
                res[start].append(BatchRes(profile_row=profile_row,
                                           inference=batch_inference,
                                           f1=batch_f1))

    return res


def filter_pf_profile(pf: List[Tuple[str, int, float]],
                      min_bw: Optional[int] = None,
                      max_bw: Optional[int] = None,
                      limit_bw_range: bool = False)\
        -> List[Tuple[str, int, float]]:
    if not limit_bw_range:
        return [profile for profile in pf if profile[0] != ""]
    else:
        if min_bw is not None and max_bw is not None:
            return [profile for profile in pf
                    if profile[0] != ""
                    and profile[1] in range(min_bw, max_bw)]
        else:
            raise ValueError


def find_base_profile(pf: List[Tuple[str, int, float]],
                      starting_max_bw: int)\
        -> Tuple[str, int, float]:
    if pf[0][1] > starting_max_bw or len(pf) == 1:
        return pf[0]
    else:
        base_profile = pf[0]
        for profile in pf:
            if profile[1] <= starting_max_bw and profile[1] >= base_profile[1]:
                base_profile = profile
        return base_profile


def gen_diff(app_idx: int, start_fid: int, end_fid: int,
             starting_max_bw: int,
             min_bw: Optional[int] = None, max_bw: Optional[int] = None,
             limit_bw_range: bool = False):

    # read all bounding boxes and ground truth in the fid range
    data_src: str = f"/home/cc/tmpfs/dds{app_idx}/workspace/merged_results"
    gt_dict: Dict[int, List[Region]] = read_offline_result(
        data_src, app_idx, True, start_fid, end_fid)
    all_res: Dict[str, Dict[int, List[Region]]] = read_all_results(
        data_src, app_idx, start_fid, end_fid)

    # calculate the inference difference vs. accuracy difference for each batch
    for end in range(start_fid + BATCH_SZ, end_fid + 1, BATCH_SZ):

        # calculate the pareto front for the batch from end - BATCH_SZ to end
        batch_pf: List[Tuple[str, int, float]] = filter_pf_profile(gen_pf(
            end_fid=end, start_bw=2000, end_bw=200000, bin_sz=5000,
            data_src=data_src, video_length=1,
            gt_dict=gt_dict, all_res=all_res, is_total=False))
        batch_pf.sort(key=lambda profile: profile[1])
        base_profile = find_base_profile(batch_pf, starting_max_bw)
        batch_pf = filter_pf_profile(batch_pf, min_bw, max_bw, limit_bw_range)

        # calculate inference difference by comparing each configuration in the
        # pf to the lowest bw-consuming profile in the pf
        for i in range(len(batch_pf)):
            inference_profile: Tuple[str, int, float] = batch_pf[i]
            base_regions: Dict[int, List[Region]] = {
                fid: all_res[base_profile[0]][fid]
                for fid in range(end - BATCH_SZ, end)
            }
            inference_regions: Dict[int, List[Region]] = {
                fid: all_res[inference_profile[0]][fid]
                for fid in range(end - BATCH_SZ, end)
            }
            diff: float = calculate_diff_concurrent(
                base_regions, inference_regions, 0.5).infer_diff
            print(end, inference_profile[1] - base_profile[1], diff,
                  inference_profile[2], base_profile[2],
                  inference_profile[2] - base_profile[2],
                  inference_profile[0][17:30], base_profile[0][17:30])


def main():
    gen_diff(2, 100, 120, 50000, 75000, 100000, True)


if __name__ == "__main__":
    main()
