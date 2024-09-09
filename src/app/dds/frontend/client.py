from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from copy import deepcopy
import json
import logging
from multiprocessing import Lock
import os
import shutil
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from api.app_server import AppServer
from api.common.enums import ResourceType
from api.pipeline_repr import InferDiff
from app.dds.dds_utils import (
    Region,
    Results,
    cleanup,
    compute_regions_size,
    merge_boxes_in_results,
)
from app.dds.scripts.util import Config, ProfileRow, calculate_diff_concurrent
from app.dds.scripts.util import Region as MyRegion


try:
    PROFILE_DIR = os.environ["PROFILE_DIR"]
except KeyError:
    PROFILE_DIR = ("/home/cc/vap-concierge/src/app/dds/"
                   "dds1/dataset/trafficcam_1")

try:
    BASELINE_MODE = bool(os.environ["BASELINE_MODE"])
except KeyError or ValueError:
    BASELINE_MODE = False

try:
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 300

try:
    BATCH_SZ = int(os.environ["BATCH_SZ"])
except KeyError or ValueError:
    BATCH_SZ = 5

try:
    APP_IDX = int(os.environ["APP_IDX"])
except KeyError:
    APP_IDX = 1

Box = List[float]
MyResults = Dict[int, List[MyRegion]]


def region_to_my_region(src: Region) -> MyRegion:
    return MyRegion(fid=src.fid, x=src.fid, y=src.y, w=src.w, h=src.h,
                    conf=src.conf, label=src.label, resolution=src.resolution,
                    origin=src.origin)


def results_to_my_results(src: Results) -> MyResults:
    return {fid: [region_to_my_region(region)
                  for region in src.regions_dict[fid]]
            for fid in src.regions_dict.keys()}


def update_dds_config(dds_config, my_config: Config):
    dds_config.low_resolution = my_config.low_res
    dds_config.low_qp = my_config.low_qp
    dds_config.high_resolution = my_config.high_res
    dds_config.high_qp = my_config.high_qp


def _callback_woker(profile_row: ProfileRow,
                    inference_config: MyResults,
                    inference_gt: MyResults,
                    iou_threshold) -> Tuple[Config, float, int]:
    f1 = 1 - calculate_diff_concurrent(
        inference_config, inference_gt, iou_threshold).infer_diff
    return profile_row.config, f1, profile_row.byte_sz


class Client(AppServer):
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""
    def __init__(self, uri: str, edge_uri: str,
                 control_port: int, hname, config, server_handle=None) -> None:
        super().__init__(uri, edge_uri, control_port)
        if hname:
            self.hname = hname
            self.session = requests.Session()
        else:
            self.server = server_handle
        self.config: Dict[str, Any] = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.logger.info("Client initialized")

        # added for resource allocator project by Roy Huang
        self.app_idx = APP_IDX
        self.profiling_start_frame: int
        self.profiling_base_inference: MyResults
        self.curr_fid: int = 0
        self.iou_threshold = 0.5
        self.my_config: Config = Config(
            low_res=self.config["low_resolution"],
            low_qp=self.config["low_qp"],
            high_res=self.config["high_resolution"],
            high_qp=self.config["high_qp"])
        self.will_backlog: bool = False
        self.p_executor = ProcessPoolExecutor()
        self.prev_start_frame: int
        self.prev_inference: MyResults = {}
        self.prev_cache_lock = Lock()
        self.data_src = PROFILE_DIR
        self.gt_dict: MyResults =\
            self.read_offline_result(True, 0, NFRAMES)
        self.all_results: Dict[str, MyResults] =\
            self.read_all_results()

    def read_all_results(self) -> Dict[str, MyResults]:
        res = {}
        exclude_files = ["profile_bw_frame", "profile_bw", "pf"]
        for file_name in os.listdir(self.data_src):
            if (file_name not in exclude_files and "gt" not in file_name):
                config_str_lst = file_name.split("_")
                low_res: float = float(config_str_lst[3])
                high_res: float = float(config_str_lst[4])
                low_qp: int = int(config_str_lst[5])
                high_qp: int = int(config_str_lst[6])
                res[file_name] = self.read_offline_result(
                    False, 0, NFRAMES, low_res, high_res, low_qp, high_qp)
        return res

    def read_offline_result(self, is_gt: bool, start_fid: int,
                            end_fid: int, low_res: Optional[float] = None,
                            high_res: Optional[float] = None,
                            low_qp: Optional[int] = None,
                            high_qp: Optional[int] = None)\
            -> MyResults:

        if high_res is not None and high_res == 1.0:
            high_res = int(high_res)

        if is_gt:
            result_path = os.path.join(self.data_src,
                                       f"trafficcam_{self.app_idx}_gt")
        else:
            result_path = os.path.join(
                self.data_src,
                f"trafficcam_{self.app_idx}_dds"
                f"_{low_res}_{high_res}_{low_qp}_{high_qp}"
                "_0.0_twosides_batch_5_0.5_0.8_0.4")
        res: MyResults = {
            fid: [] for fid in range(start_fid, end_fid)}
        with open(result_path, "r") as result_fd:
            for line in result_fd:
                row = line.split(sep=",")
                fid = int(row[0])
                if start_fid <= fid < end_fid:
                    res[fid].append(MyRegion(
                        fid, x=float(row[1]), y=float(row[2]),
                        w=float(row[3]), h=float(row[4]),
                        conf=float(row[6]), label=row[5],
                        resolution=float(row[7])
                        ))
        return res

    def read_cache_result(self, _: bool, start_fid: int,
                          end_fid: int, low_res: Optional[float] = None,
                          high_res: Optional[float] = None,
                          low_qp: Optional[int] = None,
                          high_qp: Optional[int] = None)\
            -> MyResults:

        if high_res is not None and high_res == 1.0:
            high_res = int(high_res)
        key_str: str = (f"trafficcam_{self.app_idx}_dds"
                        f"_{low_res}_{high_res}_{low_qp}_{high_qp}"
                        "_0.0_twosides_batch_5_0.5_0.8_0.4")
        config_inference: MyResults = self.all_results[key_str]
        inference: MyResults = {
            fid: config_inference[fid] for fid in range(start_fid, end_fid)
            }

        return inference

    def get_diff(self) -> InferDiff:
        if BASELINE_MODE:
            return InferDiff(-1, -1)
        if self.will_backlog:
            return InferDiff(1, 1)
        self.prev_cache_lock.acquire()
        try:
            start_fid = min(self.prev_inference.keys())
            end_fid = min(start_fid + BATCH_SZ, NFRAMES)
        except AttributeError:
            print("No Previous Inference. Cannot get profiling start frame.")
            self.prev_cache_lock.release()
            return InferDiff(1, 0)

        profiling_inference = self.read_cache_result(
                False, start_fid, end_fid, *(self.my_config.pack()))
        diff = calculate_diff_concurrent(
            profiling_inference, self.prev_inference, self.iou_threshold)
        self.prev_cache_lock.release()
        return diff

    def default_callback(self, _: bool, resource_change: float,
                         resource_t: ResourceType) -> bool:
        if resource_t == ResourceType.BW:
            self.current_bw += int(resource_change)
            print(f"Current BW {self.current_bw}")
            try:
                start_fid = min(self.curr_fid + BATCH_SZ,
                                NFRAMES)
                end_fid = min(NFRAMES, start_fid + BATCH_SZ)
            except AttributeError:
                return False

            inference_gt: MyResults = {
                    fid: self.gt_dict[fid] for fid in range(start_fid, end_fid)
                    }
            with open(os.path.join(self.data_src, "profile_bw_frame"), "r")\
                    as profile_bw_fd:
                fid_found: bool = False
                all_profile: List[ProfileRow] = []
                for line in profile_bw_fd:
                    profile_row: ProfileRow = ProfileRow(line)
                    if profile_row.start_fid == start_fid:
                        all_profile.append(profile_row)
                        fid_found = True
                    elif profile_row.start_fid != start_fid and fid_found:
                        break

            all_profile.sort(key=lambda profile_row: profile_row.byte_sz)
            if len(all_profile) == 0:
                return False
            if all_profile[0].byte_sz > self.current_bw:
                self.my_config = all_profile[0].config
                self.will_backlog = True
                return True

            max_bw: int = all_profile[len(all_profile) - 1].byte_sz
            search_max: int = min(max_bw, self.current_bw)
            # search_min: int = search_max - SEARCH_RNG
            futures_f1: List[Future] = []

            for profile_row in all_profile:
                if (profile_row.byte_sz <= search_max):
                    config = profile_row.config
                    inference_config = self.read_cache_result(
                        False, start_fid, end_fid,
                        config.low_res, config.high_res,
                        config.low_qp, config.high_qp)
                    futures_f1.append(self.p_executor.submit(
                        _callback_woker,
                        profile_row, inference_config, inference_gt,
                        self.iou_threshold
                        ))

            if len(futures_f1) >= 1:
                max_f1: float
                max_config: Config
                max_config_bw: int
                max_config, max_f1, max_config_bw = max(
                    [future.result() for future in futures_f1],
                    key=lambda result: result[1])
                self.my_config = max_config
                update_dds_config(self.config, self.my_config)
                self.will_backlog = False
                print(f"{self.container_id} changes to {self.my_config}"
                      f" f1 {max_f1} byte_sz {max_config_bw}")
                return True
            else:
                print(
                    "Cannot find satisfied configuration under current limit")
                return False
        return False

    def wait_profiling_done(self):
        # block until profiling is done
        while self.is_profiling.is_set():
            sleep(1)

    def prep_profiling(self) -> Any:
        # cache the latest inference results and video time window
        try:
            self.prev_cache_lock.acquire()
            self.profiling_start_frame = deepcopy(self.prev_start_frame)
            self.profiling_base_inference = deepcopy(self.prev_inference)
        except AttributeError:
            pass

    def update_prev_infer(self, start_frame: int, results: Results):
        self.prev_cache_lock.acquire()
        self.prev_start_frame = start_frame
        self.prev_inference = results_to_my_results(results)
        self.prev_cache_lock.release()

    def get_two_phase_results(self, start_frame, end_frame):
        req_regions = Results()
        all_required_regions = Results()
        final_results = Results()
        for fid in range(start_frame, end_frame):
            req_regions.append(Region(
                fid, 0, 0, 1, 1, 1.0, 2, self.config["low_resolution"]))
        compute_regions_size(
            req_regions, f"{self.vid_name}-profiling-base-phase",
            self.raw_images,
            self.config["low_resolution"], self.config["low_qp"],
            self.enforce_iframes, True)  # type: ignore
        results, rpn_regions = self.get_first_phase_results(
            self.vid_name, True)
        final_results.combine_results(
            results, self.config["intersection_threshold"])
        all_required_regions.combine_results(
            rpn_regions, self.config["intersection_threshold"])

        # Second Iteration
        if len(rpn_regions) > 0:
            compute_regions_size(
                rpn_regions, f"{self.vid_name}-profiling",
                self.raw_images,
                self.config["high_resolution"], self.config["high_qp"],
                self.enforce_iframes, True)  # type: ignore
            results = self.get_second_phase_results(self.vid_name, True)
            final_results.combine_results(
                results, self.config["intersection_threshold"])

        shutil.rmtree(self.vid_name + "-profiling-base-phase-cropped")
        if os.path.isdir(self.vid_name + "-profiling-cropped"):
            shutil.rmtree(self.vid_name + "-profiling-cropped")

        return final_results

    # Roy's modification ends here

    def init_server(self, nframes):
        self.config['nframes'] = nframes
        response = self.session.post(
            "http://" + self.hname + "/init", data=yaml.safe_dump(self.config))
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            # Need to add exception handling
            exit()

    def get_first_phase_results(self, vid_name, is_profiling):
        if is_profiling:
            encoded_vid_path = os.path.join(
                vid_name + "-profiling-base-phase-cropped", "temp.mp4")
            end_point = f"/profiling-low?start_id={self.profiling_start_frame}"
        else:
            encoded_vid_path = os.path.join(
                vid_name + "-base-phase-cropped", "temp.mp4")
            end_point = "/low"
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + end_point, files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config["low_resolution"], "low-res"))
        rpn = Results()
        for region in response_json["req_regions"]:
            rpn.append(Region.convert_from_server_response(
                region, self.config["low_resolution"], "low-res"))

        return results, rpn

    def get_second_phase_results(self, vid_name, is_profiling):
        if is_profiling:
            encoded_vid_path = os.path.join(
                vid_name + "-profiling-cropped", "temp.mp4")
            end_point = "/profiling-high"
        else:
            encoded_vid_path = os.path.join(
                vid_name + "-cropped", "temp.mp4")
            end_point = "/high"
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + end_point, files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config["high_resolution"], "high-res"))

        return results

    def analyze_video(
            self, vid_name, raw_images, enforce_iframes)\
            -> Tuple[Results, Tuple[int, int]]:
        final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))

        # added by Roy
        self.nframes = nframes
        self.vid_name = vid_name
        self.raw_images = raw_images
        self.enforce_iframes = enforce_iframes

        self.init_server(nframes)

        for i in range(0, nframes, self.config["batch_size"]):
            self.wait_profiling_done()
            start_frame = i
            end_frame = min(nframes, i + self.config["batch_size"])
            self.curr_fid = start_frame
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")
            seg_results = Results()

            # First iteration
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(
                    fid, 0, 0, 1, 1, 1.0, 2, self.config["low_resolution"]))
            self.wait_profiling_done()
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config["low_resolution"], self.config["low_qp"],
                enforce_iframes, True)  # type: ignore
            low_phase_size += batch_video_size
            low_qp = self.config["low_qp"]
            low_res = self.config["low_resolution"]
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                             f" Using QP {low_qp} and"
                             f" Resolution {low_res}.")
            self.wait_profiling_done()
            results, rpn_regions = self.get_first_phase_results(
                vid_name, False)
            final_results.combine_results(
                results, self.config["intersection_threshold"])
            seg_results.combine_results(
                results, self.config["intersection_threshold"])
            all_required_regions.combine_results(
                rpn_regions, self.config["intersection_threshold"])

            # Second Iteration
            if len(rpn_regions) > 0:
                self.wait_profiling_done()
                batch_video_size, _ = compute_regions_size(
                    rpn_regions, vid_name, raw_images,
                    self.config["high_resolution"], self.config["high_qp"],
                    enforce_iframes, True)  # type: ignore
                high_phase_size += batch_video_size
                high_qp = self.config["high_qp"]
                high_res = self.config["high_resolution"]
                self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                 f"phase. Using QP {high_qp} and "
                                 f"Resolution {high_res}.")
                self.wait_profiling_done()
                results = self.get_second_phase_results(vid_name, False)
                final_results.combine_results(
                    results, self.config["intersection_threshold"])
                seg_results.combine_results(
                    results, self.config["intersection_threshold"])
            self.wait_profiling_done()
            self.update_prev_infer(start_frame=start_frame,
                                   results=seg_results)

            # Cleanup for the next batch
            cleanup(vid_name, False, start_frame, end_frame)

        self.logger.info("Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config["intersection_threshold"])

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)
