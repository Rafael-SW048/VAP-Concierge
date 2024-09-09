import os
from typing import Dict, List, Tuple
from functools import reduce
from operator import add

from app.dds.dds_utils import Region
import numpy as np


try:
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 100
Box = List[float]

np.set_printoptions(  # type: ignore
    suppress=True, formatter={'float_kind': '{:4.3f}'.format},   # type: ignore
    linewidth=40)  # type: ignore


def iou(b1: Box, b2: Box) -> float:
    x1, y1, w1, h1 = [float(coord) for coord in b1]
    x2, y2, w2, h2 = [float(coord) for coord in b2]
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x4 = min(x1 + w1, x2 + w2)
    y4 = min(y1 + h1, y2 + h2)
    if x3 > x4 or y3 > y4:
        return 0
    else:
        overlap = (x4 - x3) * (y4 - y3)
        return overlap / (w1 * h1 + w2 * h2 - overlap)


def filter_results(bboxes: Dict[int, List[Region]], conf_thresh)\
        -> Dict[int, List[Region]]:

    res_dict: Dict[int, List[Region]] = {}
    for fid in bboxes.keys():
        frame_res = []
        boxes: List[Region] = bboxes[fid]
        for b in boxes:
            if (b.conf >= conf_thresh and b.w * b.h <= 0.4):
                frame_res.append(b)
        res_dict[fid] = frame_res
    return res_dict


def calculate_diff(inference_dict: Dict[int, List[Region]],
                   base_dict: Dict[int, List[Region]])\
        -> Tuple[float, List[float]]:

    f1 = 0
    false_negative: int = 0
    true_positive: int = 0
    false_positive: int = 0
    inference_dict_filtered: Dict[int, List[Region]] =\
        filter_results(inference_dict, 0.5)
    base_dict_filtered: Dict[int, List[Region]] =\
        filter_results(base_dict, 0.5)
    f1_by_fid: List[float] = [0 for _ in range(len(base_dict_filtered.keys()))]

    for fid in base_dict_filtered.keys():
        frame_false_negative: int = 0
        frame_true_positive: int = 0
        frame_false_positive: int = 0
        base_regions = base_dict_filtered[fid]
        if fid not in inference_dict_filtered.keys():
            false_negative += len(base_regions)
            f1_by_fid[fid] = 0
            continue
        infer_regions = inference_dict_filtered[fid]
        for infer_region in infer_regions:
            found = False
            for base_region in base_regions:
                intersection = iou(
                    [infer_region.x, infer_region.y,
                     infer_region.w, infer_region.h],
                    [base_region.x, base_region.y,
                     base_region.w, base_region.h])
                if intersection >= 0.5\
                        and infer_region.label == base_region.label:
                    found = True
                    break
            if found:
                true_positive += 1
                frame_true_positive += 1
            else:
                false_positive += 1
                frame_false_positive += 1

        for base_region in base_regions:
            found = False
            for infer_region in infer_regions:
                intersection = iou(
                    [infer_region.x, infer_region.y,
                     infer_region.w, infer_region.h],
                    [base_region.x, base_region.y,
                     base_region.w, base_region.h])
                if intersection >= 0.5\
                        and infer_region.label == base_region.label:
                    found = True
                    break
            if not found:
                false_negative += 1
                frame_false_negative += 1
        if (2.0 * frame_true_positive + frame_false_positive
                + frame_false_negative) == 0:
            f1 = 0
        else: 
            f1 = 2.0 * true_positive / \
                (2.0 * true_positive + false_positive + false_negative)

        f1_by_fid[fid - 50] = f1
    

    return reduce(add, f1_by_fid) / len(f1_by_fid), f1_by_fid


def parse_res(res_path: str, nframes: int) -> Dict[int, List[Region]]:
    inference: Dict[int, List[Region]] = {
        fid: [] for fid in range(50, 100)}
    with open(res_path, "r") as gt_fd:
        for line in gt_fd:
            row = line.split(sep=",")
            fid = int(row[0])
            if 50 <= fid < 100:
                inference[fid].append(Region(
                    fid, x=float(row[1]), y=float(row[2]),
                    w=float(row[3]), h=float(row[4]),
                    conf=float(row[6]), label=row[5],
                    resolution=float(row[7])
                    ))
    return inference


def main():
    dds1_res_path = ("/home/cc/vap-concierge/src/sim/dds1")
    dds1_baseline_path = ("/home/cc/vap-concierge/src/sim/dds1_baseline")
    dds1_gt_path = ("/home/cc/data-set/rene/merged_results/rene_gt")
    dds1_res = parse_res(dds1_res_path, NFRAMES)
    dds1_baseline = parse_res(dds1_baseline_path, NFRAMES)
    dds1_gt = parse_res(dds1_gt_path, NFRAMES)

    dds2_res_path = ("/home/cc/vap-concierge/src/sim/dds2")
    dds2_baseline_path = ("/home/cc/vap-concierge/src/sim/dds2_baseline")
    dds2_gt_path = ("/home/cc/data-set/uav-1/merged_results/uav-1_gt")
    dds2_res = parse_res(dds2_res_path, NFRAMES)
    dds2_baseline = parse_res(dds2_baseline_path, NFRAMES)
    dds2_gt = parse_res(dds2_gt_path, NFRAMES)

    dds1_f1_by_fid = np.array(calculate_diff(dds1_res, dds1_gt)[1])
    dds1_baseline_f1_by_fid =\
        np.array(calculate_diff(dds1_baseline, dds1_gt)[1])
    dds1_f1: float = np.mean(dds1_f1_by_fid)
    dds1_baseline_f1: float = np.mean(dds1_baseline_f1_by_fid)
    print(np.round(dds1_f1_by_fid, 5))
    print(dds1_f1, dds1_baseline_f1, dds1_f1 - dds1_baseline_f1)
    print("------------------------------------------------------------------")

    dds2_f1_by_fid = np.array(calculate_diff(dds2_res, dds2_gt)[1])
    dds2_baseline_f1_by_fid =\
        np.array(calculate_diff(dds2_baseline, dds2_gt)[1])
    dds2_f1: float = np.mean(dds2_f1_by_fid)
    dds2_baseline_f1: float = np.mean(dds2_baseline_f1_by_fid)
    print(np.round(dds2_f1_by_fid, 5))
    print(dds2_f1, dds2_baseline_f1, dds2_f1 - dds2_baseline_f1)
    print("------------------------------------------------------------------")
    improve_by_fid = dds1_f1_by_fid - dds1_baseline_f1_by_fid\
        + (dds2_f1_by_fid - dds2_baseline_f1_by_fid)
    print(np.round(improve_by_fid, 5))
    print(np.mean(improve_by_fid))
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
