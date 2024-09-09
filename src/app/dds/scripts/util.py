from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import reduce
import os
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from api.pipeline_repr import InferDiff


class DiffMode(Enum):
    F1 = 1
    Precision = 2
    Recall = 3


@dataclass
class Region():
    fid: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    label: str
    resolution: float
    origin: str = "origin"

    def __str__(self) -> str:
        return (f"{self.fid},{self.x},{self.y},{self.w},{self.h},"
                f"{self.label},{self.conf},{self.resolution},{self.origin}")

    def copy(self) -> "Region":
        return Region(self.fid, self.x, self.y, self.w, self.h, self.conf,
                      self.label, self.resolution, self.origin)


# typedef
Box = List[float]
Results = Dict[int, List[Region]]
AllResults = Dict[str, Results]


class ProfileRow():

    def __init__(self, row_str: str) -> None:
        self.str: str = row_str
        row: List[str] = row_str.split(",")
        config_str_lst: List[str] = row[0].split("_")
        self.config: Config = Config(
            float(config_str_lst[0]), float(config_str_lst[1]),
            int(config_str_lst[2]), int(config_str_lst[3]))
        self.start_fid: int = int(row[1])
        self.end_fid: int = int(row[2])
        self.byte_sz: int = int(row[3])

    def __str__(self) -> str:
        return self.str

    def __repr__(self) -> str:
        return self.str


@dataclass
class Config():
    low_res: float
    high_res: float
    low_qp: int
    high_qp: int

    def __str__(self) -> str:
        if self.high_res == 1:
            return (f"{self.low_res}_{int(self.high_res)}_{self.low_qp}_"
                    f"{self.high_qp}")
        else:
            return (f"{self.low_res}_{self.high_res}_{self.low_qp}_"
                    f"{self.high_qp}")

    def pack(self) -> Tuple[float, float, int, int]:
        return self.low_res, self.high_res, self.low_qp, self.high_qp

    def is_equal(self, that: "Config"):
        return (self.low_res == that.low_res
                and self.high_res == that.high_res
                and self.low_qp == that.low_qp
                and self.high_qp == that.high_qp)


def read_all_results(
        data_src: str, video_name: Union[str, int],
        start_fid: int, end_fid: int,
        pool: Optional[ProcessPoolExecutor] = None)\
        -> Dict[str, Dict[int, List[Region]]]:
    res = {}
    exclude_files = ["profile_bw", "profile_bw_frame"]
    print(f"Reading all results from {data_src}...")
    executor: ProcessPoolExecutor
    if pool is None:
        executor = ProcessPoolExecutor()
    else:
        executor = pool
    res_file_prefix: str
    if isinstance(video_name, int):
        res_file_prefix = f"trafficcam_{video_name}"
    else:
        res_file_prefix = video_name

    future_dict: Dict[str, Future] = {}
    for file_name in os.listdir(data_src):
        if (file_name not in exclude_files
                and "gt" not in file_name and "pf" not in file_name):

            start: int = file_name.index("dds_") + 4
            end: int = file_name.index("_0.0")
            config_str_short: str = file_name[start: end]
            config_str_lst = config_str_short.split("_")

            low_res: float = float(config_str_lst[0])
            high_res: float = float(config_str_lst[1])
            low_qp: int = int(config_str_lst[2])
            high_qp: int = int(config_str_lst[3])

            future_dict[file_name] = executor.submit(
                read_offline_result,
                data_src, res_file_prefix, False, start_fid, end_fid,
                low_res, high_res, low_qp, high_qp)

    res = {
        file_name: future.result()
        for file_name, future in future_dict.items()
        }
    print(f"Done reading all results from {data_src}")
    return res


def read_offline_result(data_src: str, video_name: Union[str, int],
                        is_gt: bool,
                        start_fid: int, end_fid: int,
                        low_res: Optional[float] = None,
                        high_res: Optional[float] = None,
                        low_qp: Optional[int] = None,
                        high_qp: Optional[int] = None)\
        -> Dict[int, List[Region]]:

    res_file_prefix: str
    if isinstance(video_name, int):
        res_file_prefix = f"trafficcam_{video_name}"
    else:
        res_file_prefix = video_name

    if high_res is not None and high_res == 1.0:
        high_res = int(high_res)

    if is_gt:
        result_path = os.path.join(data_src,
                                   f"{res_file_prefix}_gt")
    else:
        result_path = os.path.join(
            data_src,
            f"{res_file_prefix}_dds"
            f"_{low_res}_{high_res}_{low_qp}_{high_qp}"
            "_0.0_twosides_batch_5_0.5_0.8_0.4")
    res: Dict[int, List[Region]] = {
        fid: [] for fid in range(start_fid, end_fid)}
    with open(result_path, "r") as result_fd:
        for line in result_fd:
            row = line.split(sep=",")
            fid = int(row[0])
            if start_fid <= fid < end_fid:
                res[fid].append(Region(
                    fid, x=float(row[1]), y=float(row[2]),
                    w=float(row[3]), h=float(row[4]),
                    conf=float(row[6]), label=row[5],
                    resolution=float(row[7])
                    ))
    return res


def read_all_profile(data_src: str, start_fid: int) -> List[ProfileRow]:
    with open(os.path.join(data_src, "profile_bw_frame"), "r")\
            as profile_bw_fd:
        all_profile: List[ProfileRow] = []
        for line in profile_bw_fd:
            profile_row: ProfileRow = ProfileRow(line)
            if profile_row.start_fid == start_fid:
                all_profile.append(profile_row)
    all_profile.sort(key=lambda profile_row: profile_row.byte_sz)
    return all_profile


def get_byte_sz(data_src: str, config: Config, start_fid: int) -> int:
    profiles: List[ProfileRow] = read_all_profile(data_src, start_fid)
    total_bytes: int = 0
    for profile in profiles:
        if str(profile.config) == str(config):
            total_bytes += profile.byte_sz
            break
    return total_bytes


def get_gt_byte_sz(data_src: str, start_fid: int) -> int:
    with open(os.path.join(data_src, "gt_bw"), "r")\
            as profile_bw_fd:
        for line in profile_bw_fd:
            line_lst: List[str] = line.split(",")
            profile_row_start_fid: int = int(line_lst[1])
            if profile_row_start_fid == start_fid:
                return int(line_lst[3])
    return 0


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


def calculate_diff(inference_dict: Dict[int, List[Region]],
                   base_dict: Dict[int, List[Region]], iou_threshold: float)\
        -> InferDiff:

    f1: float = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    inference_dict_filtered: Dict[int, List[Region]] =\
        filter_results(inference_dict, 0.5)
    base_dict_filtered: Dict[int, List[Region]] =\
        filter_results(base_dict, 0.5)

    for fid in base_dict_filtered.keys():
        base_regions = base_dict_filtered[fid]
        if fid not in inference_dict_filtered.keys():
            false_negative += len(base_regions)
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
                if intersection >= iou_threshold\
                        and infer_region.label == base_region.label:
                    found = True
                    break
            if found:
                true_positive += 1
            else:
                false_positive += 1

        for base_region in base_regions:
            found = False
            for infer_region in infer_regions:
                intersection = iou(
                    [infer_region.x, infer_region.y,
                     infer_region.w, infer_region.h],
                    [base_region.x, base_region.y,
                     base_region.w, base_region.h])
                if intersection >= iou_threshold\
                        and infer_region.label == base_region.label:
                    found = True
                    break
            if not found:
                false_negative += 1

    if (2.0 * true_positive + false_positive + false_negative) == 0:
        print("Divider is zero")
        return InferDiff(1, 0)

    f1 = 2.0 * true_positive / (2.0 * true_positive + false_positive
                                + false_negative)
    res = 1 - f1
    return InferDiff(res, 0)


def _calculate_diff_worker(
        inference_dict_filtered_seg: Dict[int, List[Region]],
        base_dict_filtered_seg: Dict[int, List[Region]],
        iou_threshold)\
        -> Tuple[int, int, int]:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    for fid in base_dict_filtered_seg.keys():
        base_regions = base_dict_filtered_seg[fid]
        if fid not in inference_dict_filtered_seg.keys():
            false_negative += len(base_regions)
            continue
        infer_regions = inference_dict_filtered_seg[fid]
        for infer_region in infer_regions:
            found = False
            for base_region in base_regions:
                intersection = iou(
                    [infer_region.x, infer_region.y,
                     infer_region.w, infer_region.h],
                    [base_region.x, base_region.y,
                     base_region.w, base_region.h])
                if intersection >= iou_threshold\
                        and infer_region.label == base_region.label:
                    found = True
                    break
            if found:
                true_positive += 1
            else:
                false_positive += 1

        for base_region in base_regions:
            found = False
            for infer_region in infer_regions:
                intersection = iou(
                    [infer_region.x, infer_region.y,
                     infer_region.w, infer_region.h],
                    [base_region.x, base_region.y,
                     base_region.w, base_region.h])
                if intersection >= iou_threshold\
                        and infer_region.label == base_region.label:
                    found = True
                    break
            if not found:
                false_negative += 1
    return true_positive, false_positive, false_negative


def calculate_diff_concurrent(
        inference_dict: Dict[int, List[Region]],
        base_dict: Dict[int, List[Region]], iou_threshold: float,
        mode: DiffMode = DiffMode.F1,
        pool: Optional[ProcessPoolExecutor] = None)\
        -> InferDiff:

    f1: float
    agg_true_positive: int = 0
    agg_false_positive: int = 0
    agg_false_negative: int = 0
    executor: ProcessPoolExecutor
    if pool is None:
        executor = ProcessPoolExecutor(4)
    else:
        executor = pool
    inference_dict_filtered: Dict[int, List[Region]] =\
        filter_results(inference_dict, 0.5)
    base_dict_filtered: Dict[int, List[Region]] =\
        filter_results(base_dict, 0.5)
    segment_sz: int = 5

    base_dict_keys: List[int] = list(base_dict_filtered.keys())
    futures: List[Future] = []
    for start_index in range(0, len(base_dict_keys), segment_sz):
        end_index = min(start_index + segment_sz, len(base_dict_keys))
        base_dict_filtered_seg: Dict[int, List[Region]] = {
                base_dict_keys[i]: base_dict_filtered[base_dict_keys[i]]
                for i in range(start_index, end_index)
                }
        inference_dict_filtered_seg: Dict[int, List[Region]] = {
                base_dict_keys[i]: inference_dict_filtered[base_dict_keys[i]]
                for i in range(start_index, end_index)
                if base_dict_keys[i] in inference_dict_filtered.keys()
                }
        futures.append(executor.submit(
            _calculate_diff_worker,
            inference_dict_filtered_seg, base_dict_filtered_seg,
            iou_threshold))

    if len(futures) == 0:
        return InferDiff(1, 0)

    agg_true_positive, agg_false_positive, agg_false_negative = reduce(
        lambda prev_res_tuple, res_tuple:
        (prev_res_tuple[0] + res_tuple[0],
         prev_res_tuple[1] + res_tuple[1],
         prev_res_tuple[2] + res_tuple[2]),
        [future.result() for future in futures])

    divider = 2.0 * agg_true_positive + agg_false_positive + agg_false_negative
    # no object in the scene
    if divider == 0:
        return InferDiff(0, 0)

    if mode == DiffMode.F1:
        f1 = 2.0 * agg_true_positive / divider
        res = 1 - f1
    elif mode == DiffMode.Precision:
        precision = agg_true_positive /\
            (agg_true_positive + agg_false_positive)
        res = 1 - precision
    else:
        recall = agg_true_positive / (agg_true_positive + agg_false_negative)
        res = 1 - recall
    return InferDiff(res, 0)


def read_cache_result(all_results: Dict[str, Dict[int, List[Region]]],
                      app_idx: int, start_fid: int, end_fid: int,
                      low_res: Optional[float] = None,
                      high_res: Optional[float] = None,
                      low_qp: Optional[int] = None,
                      high_qp: Optional[int] = None)\
        -> Dict[int, List[Region]]:

    if high_res is not None and high_res == 1.0:
        high_res = int(high_res)
    key_str: str = (f"trafficcam_{app_idx}_dds"
                    f"_{low_res}_{high_res}_{low_qp}_{high_qp}"
                    "_0.0_twosides_batch_5_0.5_0.8_0.4")
    config_inference: Dict[int, List[Region]] = all_results[key_str]
    inference: Dict[int, List[Region]] = {
        fid: config_inference[fid] for fid in range(start_fid, end_fid)
        }

    return inference


def inference_to_str_lst(inference: Dict[int, List[Region]]) -> List[str]:
    inference_str_lst: List[str] = []
    for fid in inference.keys():
        for region in inference[fid]:
            inference_str_lst.append(str(region))
    return inference_str_lst


def copy_inference_dict(inference_dict: Dict[int, List[Region]])\
        -> Dict[int, List[Region]]:
    copy: Dict[int, List[Region]] = {}
    for fid in inference_dict.keys():
        copy[fid] = []
        for region in inference_dict[fid]:
            copy[fid].append(region.copy())
    return copy
