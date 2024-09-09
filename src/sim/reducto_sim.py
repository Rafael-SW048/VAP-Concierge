import argparse
import os
import time
from typing import NamedTuple
from threading import Thread

from api.app_server import AppServer
from api.app_server import serve as grpc_serve
from api.common.enums import ResourceType
from api.common.typedef import Any, InferCache


class Input(NamedTuple):
    seg_id: int
    is_profiling: bool


class Reducto(AppServer):

    def __init__(self, client_uri: str, need_concierge: bool,
                 control_port: int, out_file_path: str, app_idx) -> None:
        super().__init__(client_uri, need_concierge, control_port)
        self.current_bw: int
        self.seg_id: int = 0
        self.threshold = 0.5
        self.iou_threshold = 0.3
        self.app_idx = app_idx
        self.detection_file = os.path.join("/home/cc/vap-concierge/src/sim"
                                           f"/reducto-mix{self.app_idx}")
        self.out_file_path = out_file_path

    def check_accuracy(self, inference: int, gt: int) -> float:
        if gt != 0:
            return inference / gt
        else:
            return 0

    def run_inference(self, input: Input) -> int:
        time.sleep(2)
        fid_min = 30 * input.seg_id
        fid_max = fid_min + 30
        inferred_frame = 0
        print(f"Read from {fid_min} to {fid_max}")
        with open(self.detection_file, "r") as detections:
            for line in detections:
                if fid_min < int(line) < fid_max\
                        and inferred_frame < self.current_bw:
                    inferred_frame += 1
        return inferred_frame

    def default_callback(self, is_succeed: bool, actual_amount: float,
                         resource_t: ResourceType) -> Any:
        if is_succeed and resource_t == ResourceType.BW:
            self.current_bw = round(actual_amount)

    def run(self):
        while self.current_bw is None:
            pass
        while self.seg_id < 120:
            with open(self.out_file_path, "a+") as out_fd:
                input = Input(self.seg_id, False)
                detections = self.run_inference(input)
                out_fd.write(str(detections) + "\n")
                self.update_prev_iter(InferCache(
                    Input(self.seg_id, True), 0, detections))
                self.seg_id += 1


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
    dds = Reducto("", True, 5000 + args.app_idx,
                  f"/home/cc/vap-concierge/src/sim/reducto{args.app_idx}",
                  args.app_idx)
    Thread(target=grpc_serve, args=(dds,)).start()
    time.sleep(5)
    dds.checkin()
    dds.run()
