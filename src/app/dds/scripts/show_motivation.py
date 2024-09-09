import os
import sys
from typing import Dict, List

from app.dds.scripts.util import (
    ProfileRow,
    Region,
    calculate_diff,
    read_all_profile,
    read_all_results,
    read_cache_result,
    read_offline_result,
)

try: 
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 300
BATCH_SZ = 5


def main():
    data_src: str = "/home/cc/tmpfs/dds2/workspace/merged_results"
    app_idx: int = 2
    iou_threshold = 0.3
    all_res: Dict[str, Dict[int, List[Region]]] = read_all_results(
        data_src, app_idx, 0, NFRAMES)
    gt: Dict[int, List[Region]] = read_offline_result(
        data_src, app_idx, True, 0, NFRAMES)
    start_fid: int
    for start_fid in range(0, NFRAMES, BATCH_SZ):
        end_fid: int = min(NFRAMES, start_fid + BATCH_SZ)
        all_profile: List[ProfileRow] = read_all_profile(data_src, start_fid)
        max_f1: float = -1
        max_f1_bw: int = sys.maxsize
        for profile in all_profile:
            inference: Dict[int, List[Region]] = read_cache_result(
                    all_res, app_idx, start_fid, end_fid,
                    *profile.config.pack()
                    )
            batch_gt: Dict[int, List[Region]] = {
                    fid: gt[fid] for fid in range(start_fid, end_fid)
                    }
            f1: float = 1 - calculate_diff(
                inference, batch_gt,
                iou_threshold).infer_diff
            profile_bw: int = profile.byte_sz
            if f1 > max_f1:
                max_f1 = f1
                max_f1_bw = profile_bw
        print(int(end_fid / BATCH_SZ), max_f1, max_f1_bw)


if __name__ == "__main__":
    main()
