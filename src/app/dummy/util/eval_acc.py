import argparse
import functools
import operator
import os
from typing import List


def is_intersect(infer_box: List[float], gt_box: List[float],
                 iou: float) -> bool:

    # [y_min, x_min, y_max, x_max]
    # [top, left, bottom, right]
    infer_top: float = infer_box[0]
    infer_bot: float = infer_box[2]
    infer_left: float = infer_box[1]
    infer_right: float = infer_box[3]
    gt_top: float = gt_box[0]
    gt_bot: float = gt_box[2]
    gt_left: float = gt_box[1]
    gt_right: float = gt_box[3]

    if (infer_bot > gt_top and infer_top < gt_bot
            and infer_left < gt_right and infer_right > gt_left):
        intersect: List[float] = [
                max(infer_top, gt_top), max(infer_left, gt_left),
                min(infer_bot, gt_bot), min(infer_right, gt_right)
                ]

        intersect_a: float = ((intersect[2] - intersect[0])
                              * (intersect[3] - intersect[1]))
        gt_a: float = (gt_right - gt_left) * (gt_bot - gt_top)

        return intersect_a / gt_a >= iou
    return False


def get_f1(infer: List[List[float]], gt: List[List[float]], iou: float):

    if len(infer) != 0 and len(gt) != 0:
        t_positive = 0
        infer_len = len(infer)
        for _, gt_box in enumerate(gt):
            for i in range(len(infer)):
                infer_box = infer[i]
                if is_intersect(infer_box, gt_box, iou):
                    t_positive = t_positive + 1
                    del infer[i]
                    break

        if t_positive != 0:
            precision = t_positive / infer_len
            recall = t_positive / len(gt)
            return (precision * recall) / (precision + recall) * 2
    return 0


def read_result(result_filename: str) -> List[List[float]]:
    res = []
    with open(result_filename) as file:
        for line in file:
            res.append([float(coord) for coord in line.split()])
    return res


def sort_seg(frame):
    first_dash = frame.index("-")
    second_dash = frame.index("-", first_dash + 1)
    try:
        return (int(frame[first_dash + 1: second_dash]),
                int(frame[second_dash + 1:frame.index(".")]))
    except KeyError:
        print(frame)
        exit(-1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-g", "--gt_dir",
            type=str,
            required=True
            )

    parser.add_argument(
            "-i", "--infer_dir",
            type=str,
            required=True
            )
    args = parser.parse_args()
    gt_dir: str = args.gt_dir
    infer_dir = args.infer_dir

    matches = ["encoded", "DS", "low", "detection"]
    frames = [
            filename
            for filename
            in os.listdir(infer_dir)
            if (
                not any(substr in filename for substr in matches)
                and ".png" in filename
                and filename[0] != "."
                )
        ]

    frames = sorted(frames, key=sort_seg)

    prev_detection = ""
    scores: List[float] = []
    for frame in frames:
        detection = frame[:frame.index(os.path.splitext(frame)[1])]\
            + "-detection"
        gt_path = os.path.join(gt_dir, detection)

        if not os.path.exists(os.path.join(infer_dir, detection)):
            detection = prev_detection
        else:
            prev_detection = detection
        infer_path = os.path.join(infer_dir, detection)

        infer = read_result(infer_path)
        gt = read_result(gt_path)
        score = get_f1(infer, gt, 0.5)
        print(score)
        scores.append(score)

        # if os.path.exists(os.path.join(infer_dir, detection)):
        #     gt_path = os.path.join(gt_dir, detection)
        #     infer = read_result(os.path.join(infer_dir, detection))
        #     gt = read_result(gt_path)
        #     scores.append(get_f1(infer, gt, 0.5))
    print(functools.reduce(operator.add, scores) / len(scores))


if __name__ == "__main__":
    main()
