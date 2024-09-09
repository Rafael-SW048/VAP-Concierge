from functools import reduce
import logging
from operator import add, mul
import os
from time import sleep, time_ns
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pynvml import (
    c_nvmlDevice_t,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetProcessUtilization,
    nvmlInit,
)
from pynvml.nvml import NVMLError
from pynvml.nvml import c_nvmlProcessUtilizationSample_t
import torch
import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FasterRCNN

from app.dds.scripts.util import Region


# const
ONE_SEC: int = 1_000_000_000

# param
VID_DIR: str = os.getenv("VID_DIR") or "/home/cc/data-set/rene"
NFRAMES: int = int(os.getenv("NFRAMES") or "100")
BATCH_SZ: int = int(os.getenv("BATCH_SZ") or "5")
INFER_OUT: str = os.getenv("INFER_OUT") or "./infer_out"
PROFILE_OUT: str = os.getenv("PROFILE_OUT") or "./profile_out"
DIFF_THOLD: float = float(os.getenv("DIFF_THOLD") or "0.3")

# typedef
FrameInference = Dict[str, torch.Tensor]
BatchInference = Dict[int, FrameInference]


def NSEC_TO_USEC(nsec: int):
    return int(nsec / 1_000)


def TO_NUMPY(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def read_img(img_path: str) -> torch.Tensor:
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img).cuda()


def read_pids() -> List[int]:
    pids: List[int] = []
    cgroup: Optional[str] = None
    with open("/proc/self/cgroup") as cgroup_fd:
        for line in cgroup_fd:
            str_lst: List[str] = line.split(":")
            if str_lst[1] == "cpu,cpuacct":
                cgroup = str_lst[2][1:].rstrip()

    if cgroup is not None:
        with open(os.path.join("/sys/fs/cgroup/cpu", cgroup, "cgroup.procs"))\
                as procs_fd:
            for line in procs_fd:
                try:
                    pids.append(int(line))
                except ValueError:
                    pass
    else:
        pids.append(os.getpid())

    return pids


def img_diff(this: torch.Tensor, that: torch.Tensor) -> float:
    diff_tensor: torch.Tensor =\
        reduce(add, reduce(add, reduce(add, abs(this - that))))\
        / reduce(mul, this.shape)
    return diff_tensor.item()


class PixelDiffAdap():

    def __init__(self, vid_dir: str, batch_sz: int, nframes: int,
                 infer_output_path: str, profile_output_path: str,
                 diff_threshold: float):
        self.logger = logging.getLogger("GPU_ADAP_SIM")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler("./gpu_adp_sim.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-6s %(name)-14s %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        nvmlInit()
        self.model: FasterRCNN = models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True).eval().cuda()
        self.vid_dir: str = vid_dir
        self.batch_sz: int = batch_sz
        self.nframes: int = nframes
        self.infer_output_fd = open(infer_output_path, "w+")
        self.profile_output_fd = open(profile_output_path, "w+")
        self.latest_inference: Optional[
            Tuple[int, torch.Tensor, FrameInference]] = None
        self.pid: int = os.getpid()
        self.diff_threshold: float = diff_threshold
        self.dev_handle: c_nvmlDevice_t = nvmlDeviceGetHandleByIndex(0)

    def frame_filter(self, imgs: Dict[int, torch.Tensor])\
            -> Dict[int, torch.Tensor]:
        if self.latest_inference is None:
            return imgs
        else:
            final_imgs: Dict[int, torch.Tensor] = {}
            for fid, img_tensor in imgs.items():
                if img_diff(img_tensor, self.latest_inference[1])\
                        > self.diff_threshold:
                    final_imgs[fid] = img_tensor
            return final_imgs

    def padding(
            self, all_imgs: Dict[int, torch.Tensor],
            filtered_result: BatchInference) -> BatchInference:

        assert self.latest_inference is not None

        padded_res: BatchInference = {}
        # pad with the latest inference if the frame is filtered out
        for fid in all_imgs.keys():
            if fid not in filtered_result.keys():
                padded_res[fid] = self.latest_inference[2]
            else:
                padded_res[fid] = filtered_result[fid]
        return padded_res

    def write_infer_to_file(self, inference_dict: BatchInference):
        final_str: str = ""
        for fid, inference in inference_dict.items():
            boxes: torch.Tensor = inference["boxes"]
            labels: torch.Tensor = inference["labels"]
            conf_scores: torch.Tensor = inference["scores"]
            nboxes: int = len(boxes)
            for i in range(0, nboxes):
                box: np.ndarray = TO_NUMPY(boxes[i])
                region: Region = Region(
                        fid=fid,
                        x=box[0], y=box[1], w=box[2], h=box[3],
                        conf=conf_scores[i].item(),
                        label=str(labels[i].item()), resolution=1
                        )
                final_str += f"{region}\n"
        self.infer_output_fd.write(final_str)

    def wirte_profile_to_file(self, start_fid: int, end_fid: int, util: int):
        self.profile_output_fd.write(
            f"{self.diff_threshold},{start_fid},{end_fid},{util}\n")

    def run(self):
        start_fid: int = 0
        while start_fid < self.nframes:
            # read frames for this batch and filter out similar frames
            end_fid = min(start_fid + self.batch_sz, self.nframes)
            img_tensor_dict: Dict[int, torch.Tensor] = {
                fid: read_img(os.path.join(self.vid_dir, f"{fid:010}.png"))
                for fid in range(start_fid, end_fid)
                }
            final_imgs: Dict[int, torch.Tensor] = self.frame_filter(
                img_tensor_dict)

            # starting gpu profiler then run inference
            cgroup_pids: List[int] = read_pids()
            st: int = NSEC_TO_USEC(time_ns())
            inference_lst: BatchInference = {
                    fid: self.model([img_tensor])[0]
                    for fid, img_tensor in final_imgs.items()
                    }
            try:
                all_util: List[c_nvmlProcessUtilizationSample_t] =\
                    nvmlDeviceGetProcessUtilization(self.dev_handle, st)
                end: int = NSEC_TO_USEC(time_ns())
                util: int = reduce(add, [
                    proc.smUtil
                    for proc in all_util if proc.pid in cgroup_pids
                    ]) * (end - st)
            except NVMLError:
                util: int = 0

            # write gpu utilization and inference result to disk
            self.wirte_profile_to_file(start_fid, end_fid, util)
            final_res: BatchInference
            if self.latest_inference is None:
                final_res = inference_lst
            else:
                final_res = self.padding(img_tensor_dict, inference_lst)
            self.write_infer_to_file(final_res)

            # preparation for the next batch
            if len(inference_lst) != 0:
                latest_inference_fid: int = max(inference_lst.keys())
                self.latest_inference = (
                    latest_inference_fid,
                    img_tensor_dict[latest_inference_fid],
                    inference_lst[latest_inference_fid]
                    )
            self.logger.info(
                    f"Finished inference from frame {start_fid} to {end_fid}")
            start_fid += self.batch_sz
            sleep(2)


def main():
    sim: PixelDiffAdap = PixelDiffAdap(
        VID_DIR, BATCH_SZ, NFRAMES, INFER_OUT, PROFILE_OUT, DIFF_THOLD)
    sim.run()


if __name__ == "__main__":
    main()
