import logging
import os
import shutil
import requests
import json
from dds_utils import (Results, read_results_dict, cleanup, Region,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results, get_best_configuration, get_best_configuration_baseline,
                       get_bw_lower_bound, evaluate_partial_profile, readAndUpdate, writeResult)
import yaml

# VAP-Concierge Collaboration

import sys

from dds_utils import evaluate_partial, read_results_dict


from api.app_server import AppServer, serve as grpc_serve
from api.pipeline_repr import InferDiff

from api.common.enums import ResourceType
from typing import Any, Optional, Tuple

from time import sleep, perf_counter as perf_counter_s
from threading import Thread, Event
import csv
from datetime import datetime
import tempfile
from multiprocessing.pool import Pool
import concurrent.futures
# from vidgear.gears import WriteGear
# VAP-Concierge Collaboration End

class Client(AppServer):
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""

    def __init__(self, hname, config, server_handle=None,
        # VAP-Concierge Collaboration
                 app_idx: Optional[str]=None,
                 uri: Optional[str]=None,
                 edge_uri: Optional[str]=None,
                 control_port: Optional[int]=None,
                 is_faked=False):

        self.app_idx = app_idx
        self.completed_frames = 0
        self.temp_dir = None
        self.resourceChange = []

        # profiling latency measurement setup
        # if file doesn't exist
        self.profilingLatency = 0
        self.rawDelay = []
        self.prev_backend = 0
        self.prev_encoding = 0
        self.current_budget = 0.45 #default value
        # if not os.path.exists(f"../../profilingLatency-{self.app_idx}.csv"):
        #     os.system(f"touch ../../profilingLatency-{self.app_idx}.csv")
        # f = open(f"../../profilingLatency-{self.app_idx}.csv", "a+")
        # f.write("\n")
        # f.close()
        
        # sensitivity estimation #1: Offline F1 Difference
        self.prev_config_f1 = None
        self.curr_config_f1 = None
        self.vid_name = None
        self.debugBudget = []
        self.is_min_bw = False
        # VAP-Concierge Collaboration End

        # Adaptive DDS
        self.profile_no = 0
        self.real_bw = 0
        self.real_f1 = None
        # Adaptive DDS End

        # Skip Profiling Setup
        self.real_low_qp = None
        self.real_low_resolution = None
        self.real_high_qp = None
        self.real_high_resolution = None
        self.is_started = False
        self.is_allocated = False
        # End of skip profiling

        # New Baseline Setup
        self.max_bw = int(os.environ["MAX_BW"])
        self.profiling_delta = int(os.environ["PROFILING_DELTA"])
        # End of New Baseline Setup

        # Multiprocessing Setup
        self.p = Pool(5)

        if hname:
            self.hname = hname
            self.session = requests.Session()
        else:
            self.server = server_handle
        self.config = config

        self.logger = logging.getLogger(f"client{self.app_idx}")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        # VAP-Concierge Collaboration
        data_port=int(hname[hname.find(":")+1::])
        # writeResult(self.app_idx, data_port, "dataport")
        self.logger.info("Initializing AppServer")
        super().__init__(uri, edge_uri, control_port, data_port, is_faked)
        self.is_inferring = Event()

        # mark the occurance of reallocation
        self.just_reallocated = Event()

        self.logger.info("Starting gRPC Pipeline server")
        Thread(target=grpc_serve, args=(self,)).start()

        self.logger.info("Sleep for 2")
        sleep(2)

        self.logger.info("gRPC Concierge request CheckinApp")
        self.checkin()
        # VAP-Concierge Collaboration End

        self.logger.info(f"Client initialized")

    # VAP-Concierge Collaboration

    def default_callback(self, is_succeed: bool, resource_change: float, resource_t: ResourceType) -> bool:
        if (resource_t == ResourceType.BW):
            self.resource_change = resource_change

            self.change_bandwidth_and_adapt(resource_change)

            # METHOD 1: [offline] f1 difference
            # self.f1_mid = self.prev_config_f1
            # if not profiling, just skip and only pick the configuration
            if self.is_profiling.is_set():
                self.f1_mid = self.real_f1
            # if self.prev_config_f1 != None:
            #     self.f1_diff_high = self.curr_config_f1 - self.prev_config_f1 
                if self.prev_config_f1 != None:
                    # There is an inference noise
                    self.f1_diff_high = max(self.curr_config_f1 - self.real_f1, 0)
            
            return True
        else: 
            return False
        
    # change the bandwidth and get the best configuration
    def change_bandwidth_and_adapt(self, resource_change: float, skipped=False):
        # self.current_bw += resource_change
        if self.is_profiling.is_set():
            self.budget_trans += resource_change
            # writeResult(self.app_idx, self.budget_trans, "budgetDebug")
            self.debugBudget.append(self.budget_trans)
        else:
            if self.is_started:
                # self.current_bw += (1/self.current_budget) * resource_change
                self.current_bw += resource_change
            else:
                self.current_bw = resource_change
        
        self.resourceChange.append(resource_change)

        self.logger.info(f"change_bandwidth_and_adpat(): the new bandwidth limit is {self.current_bw}")
        
        if not skipped:
            try:
                self.change_to_best_configuration()
            except AttributeError:
                raise NotImplementedError(f"Cannot get the best configuration at segment {self.profile_no} with a bandwidth limit of {self.current_bw}")
    
    def prep_profiling(self) -> Any:
        self.wait_inferring_done()
        self.logger.warning(f"PROFILE START")

        self.just_reallocated.set()

        # # TODO: currently the old result is read from disk
        #self.old_results = read_results_dict(f"{self.config.profile_folder_path}/results/{self.config.real_video_name}_dds_{self.config.low_resolution}_{self.config.high_resolution}_{self.config.low_qp}_{self.config.high_qp}_{self.config.rpn_enlarge_ratio}_twosides_batch_5_0.5_0.8_0.4")

        # NEW TODO: The old result is read from the current inference result
        # self.old_results = read_results_dict(f"{self.config.profile_folder_path}/results/{self.config.real_video_name}_adaptive_{self.config.adaptive_test_display}_{self.}")

    def get_diff(self) -> InferDiff:
        # return super().get_diff()

        if self.config.baseline_mode:
            return InferDiff(infer_diff=0, latency_diff=0, infer_diff_high=0, infer_diff_low=0, bw_lower_bound=0, is_min_bw=False)

        # METHOD 1: [offline] f1 difference
        # to change the bandwidth to the lower bound
        # writeResult(self.app_idx, f"frames: {self.profile_no}, budget_trans: {self.budget_trans}, current_f1 before lower bw: {self.curr_config_f1}", "sensitivityDebug")
        self.change_bandwidth_and_adapt(-2*self.resource_change)
        # writeResult(self.app_idx, f"frames: {self.profile_no}, budget_trans: {self.budget_trans}, current_f1 after lower bw: {self.curr_config_f1}", "sensitivityDebug")
        # writeResult(self.app_idx, f"frames: {self.profile_no}, is different: {self.isConfirmed()}, is the same: {self.isConfirmed2()}", "sensitivityDebug")
        self.f1_diff_low = self.f1_mid - self.curr_config_f1
        self.f1_diff_low = 0 if self.f1_diff_low < 0 else self.f1_diff_low
        # return the bandwidth
        sanityF1 = self.curr_config_f1
        self.change_bandwidth_and_adapt(2*self.resource_change, True)
        sensitivityDebug = f"frames: {self.profile_no}, f1_mid: {self.f1_mid}, f1_real: {self.real_f1}, f1_low: {self.f1_diff_low}, f1_high: {self.f1_diff_high}, f1_low_real: {sanityF1}, current_budget: {self.current_budget}"
        # writeResult(self.app_idx, sensitivityDebug, "sensitivityDebug")
        return InferDiff(
            infer_diff = 0,
            infer_diff_high=self.f1_diff_high,
            infer_diff_low=self.f1_diff_low,
            latency_diff = 0,
            bw_lower_bound = (1/self.current_budget) * get_bw_lower_bound(f'{self.config.profile_folder_path}/{self.config.profile_folder_name}/profile-{self.profile_no}.csv'),
            is_min_bw=self.is_min_bw,
            curr_frame = self.profile_no,
            curr_f1 = self.real_f1,
            curr_budget = self.current_budget
        )
    
        # METHOD 2: [online] f1 of previous result against current result, 1-f1 (TODO: currently the new result is read from disk)

        # PROFILEING_DELTA_FRAMES = int(os.environ["PROFILEING_DELTA_FRAMES"]) # the number of frames that we use to compare

        # new_results = read_results_dict(f"{self.config.profile_folder_path}/results/{self.config.real_video_name}_{self.config.low_resolution}_{self.config.high_resolution}_{self.config.low_qp}_{self.config.high_qp}_{self.config.rpn_enlarge_ratio}_twosides_batch_5_0.5_0.8_0.4")

        # last_frame = self.end_frame-1

        # tp, fp, fn, _, _, _, f1 = evaluate_partial(
        #     min_fid = last_frame + 1 - PROFILEING_DELTA_FRAMES, 
        #     max_fid = last_frame,
        #     map_dd = self.old_results,
        #     # map_dd = self.final_results.regions_dict, # the online result
        #     map_gt = new_results,
        #     gt_confid_thresh = self.config.low_threshold, 
        #     mpeg_confid_thresh = 0.5, 
        #     max_area_thresh_gt = 0.4, 
        #     max_area_thresh_mpeg = 0.4)
        
        # self.logger.info(f"Old on New f1: {f1}")
        # self.logger.info(f"infer_diff: {1-f1}")
        
        # return InferDiff(
        #     infer_diff = 1-f1,
        #     latency_diff = 0,
        #     infer_diff_high= -1,
        #     infer_diff_low= -1
        # )

        # METHOD 3: [online] f1 of previous result against current result, 1-f1, both directions, read from previous runs
        last_frame = self.end_frame-1

        high_results = read_results_dict(f"{self.config.profile_folder_path}/results/{self.config.real_video_name}_dds_{self.config.low_resolution}_{self.config.high_resolution}_{self.config.low_qp}_{self.config.high_qp}_{self.config.rpn_enlarge_ratio}_twosides_batch_5_0.5_0.8_0.4")
        _, _, _, _, _, _, f1_high = evaluate_partial(
            min_fid = last_frame + 1 - PROFILEING_DELTA_FRAMES, 
            max_fid = last_frame,
            map_dd = self.old_results,
            # map_dd = self.final_results.regions_dict, # the online result
            map_gt = high_results,
            gt_confid_thresh = self.config.low_threshold, 
            mpeg_confid_thresh = 0.5, 
            max_area_thresh_gt = 0.4, 
            max_area_thresh_mpeg = 0.4)

        self.change_bandwidth_and_adapt(-2*self.resource_change)
        low_results = read_results_dict(f"{self.config.profile_folder_path}/results/{self.config.real_video_name}_dds_{self.config.low_resolution}_{self.config.high_resolution}_{self.config.low_qp}_{self.config.high_qp}_{self.config.rpn_enlarge_ratio}_twosides_batch_5_0.5_0.8_0.4")
        _, _, _, _, _, _, f1_low = evaluate_partial(
            min_fid = last_frame + 1 - PROFILEING_DELTA_FRAMES, 
            max_fid = last_frame,
            map_dd = low_results,
            # map_dd = self.final_results.regions_dict, # the online result
            map_gt = self.old_results,
            gt_confid_thresh = self.config.low_threshold, 
            mpeg_confid_thresh = 0.5, 
            max_area_thresh_gt = 0.4, 
            max_area_thresh_mpeg = 0.4)
        
        self.logger.info(f"f1_high: {f1_high}")
        self.logger.info(f"f1_low: {f1_low}")
        self.logger.info(f"infer_diff_high: {1- f1_high}")
        self.logger.info(f"infer_diff_low: {1- f1_low}")

        self.change_bandwidth_and_adapt(2*self.resource_change)

        return InferDiff(
            infer_diff = -1,
            infer_diff_high = 1-f1_high,
            infer_diff_low = 1-f1_low,
            latency_diff = 0
        )

    def wait_profiling_done(self):
        #block until profiling is done
        while self.is_profiling.is_set():
            #Debug
            self.logger.warning("Waiting for profile done")
            sleep(0.1)
    
    def wait_inferring_done(self):
        #block until profiling is done
        while self.is_inferring.is_set():
            #Debug
            self.logger.warning("Waiting for inference done")
            sleep(0.1)
    
    # For configuration adaptation

    def calculate_budget(self):
        remain_latency = 1 - sum([self.prev_encoding, self.prev_backend])
        # sum_latency = sum([self.prev_encoding, self.prev_backend])
        if remain_latency > 0.45 and self.is_started:
            self.current_budget = remain_latency
            return self.current_bw * self.current_budget
        else:
            self.current_budget = 0.45
            return self.current_bw * self.current_budget

    def change_to_best_configuration(self) -> bool:
        try:
            if not self.is_profiling.is_set():
                self.budget_trans = self.calculate_budget()
            low_res_best, low_qp_best, high_res_best, high_qp_best, f1_best, is_min_bw = get_best_configuration(self.budget_trans, f'{self.config.profile_folder_path}/{self.config.profile_folder_name}/profile-{self.profile_no}.csv')

            # instead of sending this result directly, we may do one inferece of dds or aws, whatever you call
            
            self.logger.info(f"old configurations: {self.config.low_resolution}, {self.config.high_resolution}, {self.config.low_qp}, {self.config.high_qp}")
            
            if low_res_best != None:
                self.config.low_qp = low_qp_best
                self.config.low_resolution = low_res_best
                self.config.high_qp = high_qp_best
                self.config.high_resolution = high_res_best
                self.f1 = f1_best

            self.prev_config_f1 = self.curr_config_f1
            self.logger.info(f"change to configurations: {self.config.low_resolution}, {self.config.high_resolution}, {self.config.low_qp}, {self.config.high_qp}")
            # run the inference here
            if self.isConfirmed():
                comparation = f"current: {self.current_bw}, real: {self.real_bw}"
                start_profiling = perf_counter_s()
                start_frame = self.profile_no * self.config.batch_size
                self.curr_config_f1 = self.profile_video(start_frame, self.config.video_name, self.config.high_images_path, self.config.enforce_iframes)
                self.profilingLatency += perf_counter_s() - start_profiling
                # readAndUpdate(self.app_idx, end_profiling, "profilingLatency")
            # meaning the configuration is the same
            elif self.isConfirmed2():
                self.curr_config_f1 = self.real_f1
            # end of the inference
            return is_min_bw

        except IndexError:
            raise AttributeError(f"Cannot get the best configuration at segment {self.profile_no} with a bandwidth limit of {self.current_bw}")
    
    # VAP-Concierge Collaboration End

    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):

        # calculate the number of frames
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        final_rpn_results = Results()
        total_size = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)

            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            req_regions = Results()

            # The entire frame is a region
            for fid in range(start_frame, end_frame):
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.low_resolution))

            # compute the video size of the batch
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{video_name}-base-phase", raw_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"{batch_video_size / 1024}KB sent "
                             f"in base phase using {self.config.low_qp}QP")

            #TODO: what does this do?
            extract_images_from_video(f"{video_name}-base-phase-cropped",
                                      req_regions)
            
            # run DNN
            results, rpn_results = (
                self.server.perform_detection(
                    f"{video_name}-base-phase-cropped",
                    self.config.low_resolution, batch_fnames))

            self.logger.info(f"Detection {len(results)} regions for "
                             f"batch {start_frame} to {end_frame} with a "
                             f"total size of {batch_video_size / 1024}KB")

            # Add results to final_results (empty Result)
            final_results.combine_results(
                results, self.config.intersection_threshold)
            
            # Add rpn_results to final_rpn_results (empty Result)
            final_rpn_results.combine_results(
                rpn_results, self.config.intersection_threshold)

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size

        # Remove regions with confid < 0.3
        # 
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(
            final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)

        return final_results, [total_size, 0]

    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False, adaptive_mode=False, bandwidth_limit_dict=None):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])

        # This is different from the self.bandwidth_limit, which is the implementation for VAP-Concierge
        profile_no = None
        bandwidth_limit = None
        if adaptive_mode:
            profile_no = 0

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)

            if (adaptive_mode):
                # If reach the next segment
                if (profile_no < len(bandwidth_limit_dict['frame_id'])):
                    if (start_fid >= bandwidth_limit_dict['frame_id'][profile_no]):
                        bandwidth_limit = bandwidth_limit_dict['bandwidth_limit'][profile_no]
                        try:
                            low_res_best, low_qp_best, high_res_best, high_qp_best = get_best_configuration(bandwidth_limit, f'{self.config.profile_folder_path}/{self.config.profile_folder_name}/profile-{profile_no}.csv')
                        except:
                            raise RuntimeError(f"Cannot get the best configuration at segment {profile_no} after frame {start_fid} with a bandwidth limit of {bandwidth_limit}. Aborting...")
                    
                        self.config.low_qp = low_qp_best
                        self.config.low_resolution = low_res_best
                        self.config.high_qp = high_qp_best
                        self.config.high_resolution = high_res_best

                        video_name = (f"results/{self.config.real_video_name}_dds_{self.config.low_resolution}_{self.config.high_resolution}_{self.config.low_qp}_{self.config.high_qp}_"
                            f"{self.config.rpn_enlarge_ratio}_twosides_batch_{self.config.batch_size}_"
                            f"{self.config.prune_score}_{self.config.objfilter_iou}_{self.config.size_obj}")
                        
                        low_results_path = f'results/{self.config.real_video_name}_mpeg_{self.config.low_resolution}_{self.config.low_qp}'
                        low_results_dict = read_results_dict(low_results_path)

                        profile_no += 1

            self.logger.info(f"Processing batch from {start_fid} to {end_fid} with parameters {self.config.low_resolution}, {self.config.low_qp}, {self.config.high_resolution}, {self.config.high_qp}")           

            # Encode frames in batch and get size
            # Make temporary frames to downsize complete frames
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                             f"in base phase")
            total_size[0] += encoded_batch_video_size

            # Low resolution phase
            low_images_path = f"{video_name}-base-phase-cropped"
            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)

            low_phase_results.combine_results(
                r1, self.config.intersection_threshold)
            final_results.combine_results(
                r1, self.config.intersection_threshold)

            # High resolution phase
            if len(req_regions) > 0:
                # Crop, compress and get size
                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")
                total_size[1] += regions_size

                # High resolution phase every three filter
                r2 = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions)
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")

                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                final_results.combine_results(
                    r2, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        if (adaptive_mode):
            video_name = (f"results/{self.config.real_video_name}_adaptive_{self.config.adaptive_test_display}_"
                            f"{self.config.rpn_enlarge_ratio}_twosides_batch_{self.config.batch_size}_"
                            f"{self.config.prune_score}_{self.config.objfilter_iou}_{self.config.size_obj}")

        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        return final_results, total_size

    def init_server(self, nframes):
        self.config['nframes'] = nframes
        response = self.session.post(
            "http://" + self.hname + "/init", data=yaml.dump(self.config))
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            # Need to add exception handling
            exit()

    def get_first_phase_results(self, vid_name, start_frame=None):
        """ POST first phase, and get the result
        """
        encoded_vid_path = os.path.join(
            vid_name + "-base-phase-cropped", "temp.mp4")
        payload = {"start_frame": start_frame if start_frame != None else -1}
        video_to_send = {"json": (None, json.dumps(payload), 'application/json'), "media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/low", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        rpn = Results()
        for region in response_json["req_regions"]:
            rpn.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        backendTime = response_json["backendTime"]
        inferTime = response_json["infer_delay"]
        decode_time = response_json["end_decode"]

        return results, rpn, inferTime, backendTime, decode_time

    def get_second_phase_results(self, vid_name):
        """ POST second phase, and get the result
        """
        encoded_vid_path = os.path.join(vid_name + "-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/high", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.high_resolution, "high-res"))
        # writeResult(self.app_idx, results, "secondIterDebug")
        backendTime = response_json["backendTime"]
        inferTime = response_json["infer_delay"]
        decode_time = response_json["end_decode"]

        return results, inferTime, backendTime, decode_time

    def analyze_video(
            self, vid_name, raw_images, config, enforce_iframes, profile_info):
        self.fix_bw = self.current_bw
        self.final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))
        segmentalF1Score = []
        self.ground_truth_dict = read_results_dict(self.config.ground_truth)

        # VAP-Concierge Collaboration
        self.profile_segment_size = profile_info['segment_size']
        # VAP-Concierge Collaboration End

        self.init_server(nframes)

        # Delay Measurement Setup
        e2eStartTime = perf_counter_s()
        encodingDelay = 0
        transmissionDelay = 0
        inferenceDelay = 0
        optimizeAndReallocDelay = 0
        tempTime = 0
        tempTime2 = 0
        inferDelay = []
        backendDelay = 0
        endReading = 0
        decode_time = 0
        # Delay Measurement Setup End

        # # Let the concierge determine the bw needed for every app
        if self.config.baseline_mode or True:
            # assume bw now is ranging from 200 to 600 and hardcoded
            # get the average, of course keep considering the budget
            # bw_list = [200, 300, 400, 500, 600]
            # bw_list = [200, 350, 500]
            middle_bw = self.max_bw//3
            step = 10
            bw_list = [i for i in range(middle_bw-self.profiling_delta, middle_bw+self.profiling_delta+step, step)]
            # bw_list = [self.curr]
            f1_score_10 = [0 for i in range(len(bw_list))]
            total_seg = 10
            for bw in enumerate(bw_list):
                f1_temp = 0
                for i in range(total_seg):
                    _, _, _, _, f1_best, is_min_bw = get_best_configuration_baseline(self.current_budget*bw[1], f'{self.config.profile_folder_path}/{self.config.profile_folder_name}/profile-{i}.csv')
                    # writeResult(self.app_idx, is_min_bw, "testMinBW")
                    f1_temp += f1_best if not is_min_bw else 0
                f1_score_10[bw[0]] = f1_temp/total_seg
            # writeResult(self.app_idx, f1_score_10, "sanityCheckF1")
            status = self.get_best_bw_baseline(f1_score_10)
            while not self.is_allocated:
                sleep(0.1)

        writeResult(self.app_idx, self.current_bw, "sanityCheckBaseline")
        self.is_started = True
        for i in range(0, nframes, self.config.batch_size):
            self.temp = Results() # to obtain current f1-score
            all_required_regions_temp = Results()

            # VAP-Concierge Collaboration
            st = perf_counter_s()
            self.wait_profiling_done()
            if self.should_end_gpu_task.is_set():
                break
            self.is_inferring.set()
            self.real_bw = self.current_bw
            # VAP-Concierge Collaboration End

            start_frame = i
            # if there is a reallocation, it would start from here.
            # the point is, if there is a profiling, the whole processes would be blocked
            # I couldn't place the transmission delay counter here.
            # If baseline method is true, this would be bypassed
            if self.real_low_qp == None:
                self.real_low_qp = self.config.low_qp
                self.real_low_resolution = self.config.low_resolution
                self.real_high_qp = self.config.high_qp
                self.real_high_resolution = self.config.high_resolution
            # VAP-Concierge Collaboration
            
            if (start_frame//self.profile_segment_size > self.profile_no):
                # need to load the next profile
                self.profile_no = start_frame//self.profile_segment_size
                try:
                    self.is_min_bw = self.change_to_best_configuration()
                    # at every iteration, the current configuration should be stored
                    self.real_low_qp = self.config.low_qp
                    self.real_low_resolution = self.config.low_resolution
                    self.real_high_qp = self.config.high_qp
                    self.real_high_resolution = self.config.high_resolution

                except AttributeError:
                    raise NotImplementedError(f"Cannot get the best configuration at segment {self.profile_no} with a bandwidth limit of {self.current_bw}")
            
            
            self.log_bandwidth_limit(start_frame, self.current_bw, self.just_reallocated.is_set())
            if self.just_reallocated.is_set():
                finishRealloc = datetime.now()
                self.just_reallocated.clear()
            # VAP-Concierge Collaboration End

            self.end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {self.end_frame}")

            debugTime = 0
            # First iteration
            req_regions = Results()
            for fid in range(start_frame, self.end_frame):
                req_regions.append(Region(
                    fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            encodingStart = perf_counter_s()
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True, self.p)
            encodingDelay += perf_counter_s() - encodingStart
            self.prev_encoding = (perf_counter_s() - encodingStart)
            low_phase_size += batch_video_size
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                             f"Using QP {self.config.low_qp} and "
                             f"Resolution {self.config.low_resolution}.")
            transmissionStart = perf_counter_s()
            results, rpn_regions, inferTempFirst, backendTime, decode_temp = self.get_first_phase_results(vid_name)
            transmissionDelay += perf_counter_s() - transmissionStart - backendTime
            backendDelay += backendTime
            decode_time += decode_temp
            self.prev_backend = backendTime
            self.final_results.combine_results(
                results, self.config.intersection_threshold)
            self.temp.combine_results(
                results, self.config.intersection_threshold)
            all_required_regions_temp.combine_results(
                rpn_regions, self.config.intersection_threshold)
            all_required_regions.combine_results(
                rpn_regions, self.config.intersection_threshold)

            # Second Iteration
            inferTempSec = 0
            if len(rpn_regions) > 0:
                encodingStart = perf_counter_s()
                batch_video_size, _ = compute_regions_size(
                    rpn_regions, vid_name, raw_images,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True, self.p)
                encodingDelay += perf_counter_s() - encodingStart
                self.prev_encoding += (perf_counter_s() - encodingStart)
                high_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                 f"phase. Using QP {self.config.high_qp} and "
                                 f"Resolution {self.config.high_resolution}.")
                transmissionStart = perf_counter_s()
                results, inferTempSec, backendTime, decode_temp = self.get_second_phase_results(vid_name)
                decode_time += decode_temp
                transmissionDelay += perf_counter_s() - transmissionStart - backendTime
                backendDelay += backendTime
                self.final_results.combine_results(
                    results, self.config.intersection_threshold)
                self.temp.combine_results(
                results, self.config.intersection_threshold)
                self.prev_backend += backendTime
            inferDelay.append(inferTempFirst + inferTempSec)
            startReading = perf_counter_s()
            self.logger.info(f"Merging results for partial evaluation....")
            self.temp = merge_boxes_in_results(
            self.temp.regions_dict, 0.3, 0.3)

            self.temp.combine_results(
            all_required_regions_temp, self.config.intersection_threshold)
            
            self.logger.info("Reading ground truth results complete")
            _, _, _, _, _, _, self.curr_config_f1 = evaluate_partial(start_frame,
                self.end_frame - 1, self.temp.regions_dict, self.ground_truth_dict,
                self.config.low_threshold, 0.5, 0.4, 0.4)
            endReading += perf_counter_s() - startReading
            segmentalF1Score.append(self.curr_config_f1)
            # previous f1 score needs to be being reset at every iteration since we only consider current segment
            # self.prev_config_f1 = None
            self.real_f1 = self.curr_config_f1
            # VAP-Concierge Collaboration
            # add rpn to the current final_results for online comparison
            # self.final_results.combine_results(
            # all_required_regions, self.config.intersection_threshold)
            # VAP-Concierge Collaboration Ends

            # Cleanup for the next batch
            self.rawDelay.append(debugTime)
            # writeResult(self.app_idx, debugTime, "rawDelay")
            cleanup(vid_name, False, start_frame, self.end_frame)

            # VAP-Concierge Collaboration
            self.completed_frames = self.end_frame
            self.is_inferring.clear()
            # VAP-Concierge Collaboration End

        # VAP-Concierge Collaboration
        if not self.config.baseline_mode and not self.should_end_gpu_task.is_set():
            # notify the server that the inference is done, and to end all other inferences
            self.done_inference()
        # VAP-Concierge Collaboration End

        # if this app is finished first, set the flag as well
        e2eTime = perf_counter_s() - e2eStartTime
        self.should_end_gpu_task.set()

        self.logger.info(f"Merging results")
        self.final_results = merge_boxes_in_results(
            self.final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        self.final_results.fill_gaps(nframes)

        self.final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        self.final_results.write(f"{vid_name}")

        writeResult(self.app_idx, f"{self.config.real_video_name} - {self.fix_bw * 3}", "sanityCheck")
        writeResult(self.app_idx, self.debugBudget, "budget")
        writeResult(self.app_idx, self.resourceChange, "resource")
        writeResult(self.app_idx, endReading, "endReading")
        writeResult(self.app_idx, e2eTime, "e2eDelay")
        writeResult(self.app_idx, encodingDelay, "encodingDelay")
        writeResult(self.app_idx, transmissionDelay, "transmissionDelay")
        writeResult(self.app_idx, decode_time, "decodeTime")
        writeResult(self.app_idx, inferDelay, "inferenceDelay")
        writeResult(self.app_idx, backendDelay, "backendDelay")
        writeResult(self.app_idx, self.completed_frames, "completedFrames")
        f = open(f"../../segmentalF1Score-{self.app_idx}.csv", "a")
        with f:
            writer = csv.writer(f,delimiter=",")
            writer.writerow(segmentalF1Score)
        # f.write(str(array) + '\n')
        f.close()

        return self.final_results, (low_phase_size, high_phase_size)
    
    # VAP-Concierge Collaboration

    def profile_video(
            self, start_frame, vid_name, raw_images, enforce_iframes):
        self.profile_results = Results()
        all_required_regions_profile = Results()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))

        # VAP-Concierge Collaboration
        self.logger.info(f"Profiling frames {start_frame} to {self.end_frame}")
        # First iteration
        req_regions = Results()
        for fid in range(start_frame, self.end_frame):
            req_regions.append(Region(
                fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        batch_video_size, _ = compute_regions_size(
            req_regions, f"{vid_name}-base-phase", raw_images,
            self.config.low_resolution, self.config.low_qp,
            enforce_iframes, True, self.p)
        self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                            f"Using QP {self.config.low_qp} and "
                            f"Resolution {self.config.low_resolution}.")
        results, rpn_regions, _, _, _ = self.get_first_phase_results(vid_name, start_frame)
        self.profile_results.combine_results(
            results, self.config.intersection_threshold)
        all_required_regions_profile.combine_results(
            rpn_regions, self.config.intersection_threshold)

        # Second Iteration
        if len(rpn_regions) > 0:
            batch_video_size, _ = compute_regions_size(
                rpn_regions, vid_name, raw_images,
                self.config.high_resolution, self.config.high_qp,
                enforce_iframes, True, self.p)
            self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                f"phase. Using QP {self.config.high_qp} and "
                                f"Resolution {self.config.high_resolution}.")
            results, _, _, _ = self.get_second_phase_results(vid_name)
            self.profile_results.combine_results(
                results, self.config.intersection_threshold)

            # Cleanup for the next batch
        cleanup(vid_name, False, start_frame, self.end_frame)

        self.profile_results = merge_boxes_in_results(
            self.profile_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Getting results for {vid_name}")

        self.profile_results.combine_results(
            all_required_regions_profile, self.config.intersection_threshold)


        # this should return only f1-score
        # ground_truth_dict = read_results_dict(self.config.ground_truth)
        self.logger.info("Reading ground truth results complete")
        tp, fp, fn, _, _, _, f1 = evaluate_partial(start_frame,
            self.end_frame - 1, self.profile_results.regions_dict, self.ground_truth_dict,
            self.config.low_threshold, 0.5, 0.4, 0.4)

        return f1

    def log_bandwidth_limit(self, frame_id, bandwidth_limit, just_reallocated):
        log_directory = "logs"

        if not os.path.exists(log_directory):
            os.mkdir(log_directory)

        self.logger.info("Logging bandwidth info")

        with open(f"{log_directory}/bandwidth_log", "a") as f:
            f.write(f"{frame_id},{bandwidth_limit},{self.config.low_resolution},{self.config.high_resolution},{self.config.low_qp},{self.config.high_qp},{just_reallocated}\n")

    def isConfirmed(self):
        if (self.real_high_qp != self.config.high_qp or self.real_high_resolution != self.config.high_resolution or self.real_low_qp != self.config.low_qp or self.real_low_resolution != self.config.low_resolution):
            return self.is_profiling.is_set() and not self.should_end_gpu_task.is_set()
    
    def isConfirmed2(self):
        if (self.real_high_qp == self.config.high_qp and self.real_high_resolution == self.config.high_resolution and self.real_low_qp == self.config.low_qp and self.real_low_resolution == self.config.low_resolution):
            return self.is_profiling.is_set() and not self.should_end_gpu_task.is_set()
    # VAP-Concierge Collaboration Ends
    #self.current_bw != self.real_bw