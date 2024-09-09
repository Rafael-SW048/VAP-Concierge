import logging
import os
from threading import Thread
from time import CLOCK_REALTIME, clock_gettime_ns, perf_counter_ns, sleep
import random
from typing import List

import cv2
import numpy as np
import pynvml
import torch
import torchvision
from torchvision import models
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)

from api.app_server import AppServer
from api.app_server import serve as grpc_serve
from api.common.enums import ResourceType
from api.pipeline_repr import InferDiff

CTRL_PRT: int = int(os.getenv("CTRL_PRT") or 5001)
BATCH_SZ: int = int(os.getenv("BATCH_SZ") or 1)
DATA_SET: str = os.getenv("DATA_SET") or "/home/cc/data-set/rene"
PUBLIC_IP: str = os.getenv("PUBLIC_IP") or "127.0.0.1"
HOSTNAME: str = os.getenv("HOSTNAME") or "SCHED_TESTER"

logger = logging.getLogger(HOSTNAME)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f"./{HOSTNAME.lower()}.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)-6s %(name)-14s %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)


def gpu_monitor(handle):
    while True:
        sleep(0.1)
        logger.debug(
            "GPU utilization"
            f" {pynvml.nvmlDeviceGetUtilizationRates(handle).gpu}")


class SchedulerTester(AppServer):
    def __init__(self, uri: str, edge_uri: str, control_port: int) -> None:
        super().__init__(uri, edge_uri, control_port)
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(
                weights=self.weights).eval().cuda()
        self.model_preprocess = self.weights.transforms()
        self.fid: int = 0
        img_path = f"/home/cc/data-set/rene/{self.fid:010}.png"
        self.img: torch.Tensor =\
            self.model_preprocess(read_img(img_path)).unsqueeze(0).cuda()

    def default_callback(self, *_) -> bool:
        return True

    def prep_profiling(self) -> None:
        return None

    def get_diff(self) -> InferDiff:
        return InferDiff(infer_diff=0, latency_diff=0)

    # @AppServer._infer_wrapper
    def infer(self):
        # logger.debug("start infer")
        self.imgs: List[torch.Tensor] = [
            self.model_preprocess(read_img(os.path.join(
                DATA_SET, f"{int(random.randrange(0, 100)):010}.png"))).cuda()
            for _ in range(0, BATCH_SZ)]
        st: int = perf_counter_ns()
        res: torch.Tensor = self.model(self.imgs)
        logger.debug(f"{clock_gettime_ns(CLOCK_REALTIME)}"
                     f" infer latency: {perf_counter_ns() - st}")
        return res

    def run(self):
        self.infer()
        self.infer()
        self.infer()
        self.infer()
        self.infer()

def main():
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # Thread(target=gpu_monitor, args=(handle,)).start()

    tester: SchedulerTester = SchedulerTester(
            uri=PUBLIC_IP, edge_uri=PUBLIC_IP, control_port=CTRL_PRT)
    Thread(target=grpc_serve, args=(tester,)).start()
    sleep(1)

    tester.checkin()
    tester.run()


if __name__ == "__main__":
    main()
