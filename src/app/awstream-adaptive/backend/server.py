import os
import shutil
import logging
import cv2 as cv
from dds_utils import (Results, Region, calc_iou, merge_images, merge_images_multi, merge_images_sub,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict, writeResult, get_regions_to_query_multi, combine_regions)
from .object_detector import Detector
from multiprocessing.pool import Pool
from threading import Thread
import concurrent.futures
from time import perf_counter as perf_counter_s


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, config, nframes=None):
        self.config = config

        app_idx = os.environ["APP_IDX"]
        self.logger = logging.getLogger(f"server{app_idx}")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        file_handler = logging.FileHandler("./server.log")
        self.read_duration = int(os.environ["READ_DURATION"])
        self.delta_step = int(os.environ["DELTA_STEP"])
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        self.detector = Detector()

        self.curr_fid = self.read_duration*self.config.batch_size
        self.nframes = nframes
        self.last_requested_regions = None

        self.logger.info("Server started")
        self.p = Pool(5)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    def reset_state(self, nframes):
        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped", f))

    def perform_server_cleanup(self):
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped", f))

    def server_infer(self,image):
        detection_results, rpn_results, infer_temp = self.detector.infer(image)
        return detection_results, rpn_results, infer_temp

    def perform_detection(self, images_direc, resolution, fnames=None,
                          images=None):
        final_results = Results()
        second_phase = False
        rpn_regions = Results()
        imgs = []
        fids = []
        # fids = []
        # # images = []
        # ioJobs = []
        resolutions = []
        paths = []
        min_object_sizes = []
        # detectors = []

        # if fnames is None:
        #     fnames = sorted(os.listdir(images_direc))
        # self.logger.info(f"Running inference on {len(fnames)} frames")

        # for fname in fnames:
        #     fid = int(fname.split(".")[0])
        #     fids.append(fid)
        #     resolutions.append(resolution)
        #     min_object_sizes.append(self.config.min_object_size)
        #     detectors.append(server)
        #     image = None
        #     if images:
        #         image = images[fid]
        #     else:
        #         image_path = os.path.join(images_direc, fname)
        #         paths.append(image_path)
        #         # image = cv.imread(image_path)
        #     # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # # Reading images concurrently
        # images = []
            
        # detection_results, rpn_results, infer_temp = self.detector.infer(images)

        # with Pool(5) as p:
        #     imagesZipped = zip(fids, images, resolutions, min_object_sizes, detectors)
        #     results = p.starmap(detectImage, (imagesZipped))

        # iteration = 0
        # for result in results:
        #     final_results.append(result[iteration][0])
        #     rpn_regions.append(result[iteration][1])
        #     iteration += 1
        

        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            fids.append(fid)
            resolutions.append(resolution)
            min_object_sizes.append(self.config.min_object_size)
            image = None
            if images:
                image = images[fid]
                imgs.append(image)
            else:
                image_path = os.path.join(images_direc, fname)
                paths.append(image_path)
                # image = cv.imread(image_path)
            # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # imgs.append(image)
        if images:
            imgs = [cv.cvtColor(image, cv.COLOR_BGR2RGB) for image in imgs]
        else:
            processes = [self.executor.submit(cv.imread, path) for path in paths]
            imgs = [cv.cvtColor(f.result(), cv.COLOR_BGR2RGB) for f in processes]
        # startOpen = perf_counter_s()
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])
        # startEnd = perf_counter_s() - startOpen
        # writeResult(appNum, startEnd, "openDebug")
        if images:
            second_phase = True
        # startBatch = perf_counter_s()
        detection_results, rpn_results, infer_temp = (
            self.detector.infer(imgs, second_phase, self.p))
        # endBatch = perf_counter_s() - startBatch
        # writeResult(appNum, endBatch, "inferDebugBatch")
    
        # this process can be paralellised
        startTime = perf_counter_s()
        # if images:
        #     rpn_results = [[] for i in range(5)]
        # argsZipped = list(zip(fids, detection_results, rpn_results, min_object_sizes, resolutions))
        # results = self.p.starmap(combine_regions, (argsZipped))
        
        
        
        # writeResult(appNum, results, "resultsDebug")
        
        # for iteration in range(len(results)):
        #     final_results.append_batch(results[iteration][0], fids[iteration])
        #     if not images:
        #         rpn_regions.append_batch(results[iteration][1], fids[iteration])
        
        for i in range(len(fids)):
            frame_with_no_results = True
            # infer_delay += infer_temp
            for label, conf, (x, y, w, h) in detection_results[i]:
                if (self.config.min_object_size and
                        w * h < self.config.min_object_size) or w * h == 0.0:
                    continue
                r = Region(fids[i], x, y, w, h, conf, label,
                            resolution, origin="mpeg")
                final_results.append(r)
                frame_with_no_results = False
            for label, conf, (x, y, w, h) in rpn_results[i]:
                r = Region(fids[i], x, y, w, h, conf, label,
                            resolution, origin="generic")
                rpn_regions.append(r)
                frame_with_no_results = False
            self.logger.debug(
                f"Got {len(final_results)} results "
                f"and {len(rpn_regions)} for {fname}")

            if frame_with_no_results:
                final_results.append(
                    Region(fids[i], 0, 0, 0, 0, 0.1, "no obj", resolution))
        endTime = perf_counter_s() - startTime
        # writeResult(appNum, endTime, "multiProsDelay")

        return final_results, rpn_regions, infer_temp
        
    def get_regions_to_query(self, rpn_regions, detections): 
        req_regions = Results()
        for region in rpn_regions.regions:
            # Continue if the size of region is too large
            if region.w * region.h > self.config.size_obj:
                continue

            # If there are positive detections and they match a region
            # skip that region
            if len(detections) > 0:
                matches = 0
                for detection in detections.regions:
                    if (calc_iou(detection, region) >
                            self.config.objfilter_iou and
                            detection.fid == region.fid and
                            region.label == 'object'):
                        matches += 1
                    if matches > 0:
                        break
                if matches > 0:
                    continue

            # Enlarge and add to regions to be queried
            region.enlarge(self.config.rpn_enlarge_ratio)
            req_regions.add_single_result(
                region, self.config.intersection_threshold)
        return req_regions

    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, simulation=True,
                           rpn_enlarge_ratio=0.0, extract_regions=True):
        if extract_regions:
            # If called from actual implementation
            # This will not run
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            extract_images_from_video(images_direc, base_req_regions)

        batch_results = Results()

        self.logger.info(f"Getting results with threshold "
                         f"{self.config.low_threshold} and "
                         f"{self.config.high_threshold}")
        # Extract relevant results
        # divided by its fid
        # argsZipped = zip(fids, results_dict)
        # processes = self.p.starmap(simulate_multi, argsZipped)
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])  
        # startBatch = perf_counter_s()
        for fid in range(start_fid, end_fid):

            if fid not in results_dict:
                results_dict[fid] = []
                
            fid_results = results_dict[fid]
            # batch_results.add_multiple_results(fid_results, fid, self.config.intersection_threshold)
            for single_result in fid_results:
                single_result.origin = "low-res"
                batch_results.add_single_result(
                    single_result, self.config.intersection_threshold)
        # endBatch = perf_counter_s() - startBatch
        # writeResult(appNum, endBatch, "simulatePart1")

        detections = Results()
        rpn_regions = Results()
        # Divide RPN results into detections and RPN regions
        # cpu-bound?
        # startBatch = perf_counter_s()
        for single_result in batch_results.regions:
            if (single_result.conf > self.config.prune_score and
                    single_result.label == "vehicle"):
                detections.add_single_result(
                    single_result, self.config.intersection_threshold)
            else:
                rpn_regions.add_single_result(
                    single_result, self.config.intersection_threshold)
        # endBatch = perf_counter_s() - startBatch
        # writeResult(appNum, endBatch, "simulatePart2")

        startBatch = perf_counter_s()
        rpn_regions_list = []
        detections_list = []
        objfilter_iou_list = []
        rpn_enlarge_ratio_list = []
        intersection_threshold_list = []
        size_obj_list = []
        for fid in range(start_fid, end_fid):
            rpn_regions_list.append(rpn_regions.regions_dict[fid] if fid in rpn_regions.regions_dict else [])
            detections_list.append(detections.regions_dict[fid] if fid in detections.regions_dict else [])
            objfilter_iou_list.append(self.config.objfilter_iou)
            rpn_enlarge_ratio_list.append(self.config.rpn_enlarge_ratio)
            intersection_threshold_list.append(self.config.intersection_threshold)
            size_obj_list.append(self.config.size_obj)
        argsZipped = list(zip(rpn_regions_list, detections_list, objfilter_iou_list, rpn_enlarge_ratio_list, intersection_threshold_list, size_obj_list))
        processes = self.p.starmap(get_regions_to_query_multi, argsZipped)
        regions_to_query = Results()
        for result in processes:
            regions_to_query.combine_results(result, self.config.intersection_threshold)
        # rpn_regions, detections, objfilter_iou, rpn_enlarge_ratio, intersection_threshold, size_obj
        # regions_to_query = self.get_regions_to_query(rpn_regions, detections)
        # endBatch = perf_counter_s() - startBatch
        # writeResult(appNum, endBatch, "simulatePart3")

        return detections, regions_to_query

    def emulate_high_query(self, vid_name, low_images_direc, req_regions):
        images_direc = vid_name + "-cropped"
        # Extract images from encoded video
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])   

        startDecodingHigh = perf_counter_s()
        extract_images_from_video(images_direc, req_regions, self.executor)
        endDecodingHigh = perf_counter_s() - startDecodingHigh
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])
        # writeResult(appNum, endDecodingHigh, "decodingHighDelay")
        # writeResult(appNum, endDecodingHigh, "decodingHighDelay")
        
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])
        # for r in req_regions.regions:
        #     writeResult(appNum, r, "regionsDebug")

        if not os.path.isdir(images_direc):
            self.logger.error("Images directory was not found but the "
                              "second iteration was called anyway")
            return Results()

        fnames = sorted([f for f in os.listdir(images_direc) if "png" in f])

        # Make seperate directory and copy all images to that directory
        merged_images_direc = os.path.join(images_direc, "merged")
        os.makedirs(merged_images_direc, exist_ok=True)
        for img in fnames:
            shutil.copy(os.path.join(images_direc, img), merged_images_direc)
        # startEnd = perf_counter_s() - startOpen
        # writeResult(appNum, startEnd, "openDebug")
        # merged_images = merge_images(
        #     merged_images_direc, low_images_direc, req_regions)
        merged_images = merge_images_multi(
            merged_images_direc, low_images_direc, req_regions, self.p)

    
        # endMergeHigh = perf_counter_s() - startMergeHigh
        # writeResult(appNum, endMergeHigh, "mergeHighDelay")
        # startPerformHigh = perf_counter_s()
        results, _, infer_delay = self.perform_detection(
            merged_images_direc, self.config.high_resolution, fnames,
            merged_images)
        # endPerformHigh = perf_counter_s() - startPerformHigh - infer_delay
        # writeResult(appNum, endPerformHigh, "performHighDelay")
        results_with_detections_only = Results()
        # argsZipped = zip()
        # processes = self.p.starmap(detectionResults, argsZipped)
        # startAddSingleHigh = perf_counter_s()
        for r in results.regions:
            if r.label == "no obj":
                continue
            results_with_detections_only.add_single_result(
                r, self.config.intersection_threshold)
        # endAddSingleHigh = perf_counter_s() - startAddSingleHigh
        # writeResult(appNum, endAddSingleHigh, "addHighDelay")

        # high_only_results = Results()
        # area_dict = {}
        # for r in results_with_detections_only.regions:
        #     frame_regions = req_regions.regions_dict[r.fid]
        #     regions_area = 0
        #     if r.fid in area_dict:
        #         regions_area = area_dict[r.fid]
        #     else:
        #         regions_area = compute_area_of_frame(frame_regions)
        #         area_dict[r.fid] = regions_area
        #     regions_with_result = frame_regions + [r]
        #     total_area = compute_area_of_frame(regions_with_result)
        #     extra_area = total_area - regions_area
        #     if extra_area < 0.05 * calc_area(r):
        #         r.origin = "high-res"
        #         high_only_results.append(r)
        shutil.rmtree(merged_images_direc)
        # t = Thread(target=shutil.rmtree, args=(merged_images_direc,))
        # t.start()

        return results_with_detections_only, infer_delay, endDecodingHigh

    def perform_low_query(self, vid_data, start_fid=None):
        # Write video to file
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3]) 
        # startReceive = perf_counter_s()
        with open(os.path.join("server_temp", "temp.mp4"), "wb") as f:
            f.write(vid_data.read())
        # endReceive = perf_counter_s() - startReceive
        # writeResult(appNum, endReceive, "receiveDelay")
        

        # Extract images
        # Make req regions for extraction
        process = "Processing" if start_fid == None else "Profiling"
        start_fid = self.curr_fid if start_fid == None else start_fid
        end_fid = min(start_fid + self.config.batch_size, self.nframes)
        self.logger.info(f"{process} frames from {start_fid} to {end_fid}")
        req_regions = Results()
        for fid in range(start_fid, end_fid):
            req_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))  
        startDecode = perf_counter_s() 
        extract_images_from_video("server_temp", req_regions, self.executor)
        end_decode = perf_counter_s() - startDecode
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])
        # writeResult(appNum, end_decode, "decodingDelay")
        fnames = [f for f in os.listdir("server_temp") if "png" in f]

        # startPerform = perf_counter_s()
        results, rpn, infer_delay = self.perform_detection(
            "server_temp", self.config.low_resolution, fnames)
        # endPerform = perf_counter_s() - startPerform - infer_delay
        # writeResult(appNum, endPerform, "performDelay")
        # # appNum = os.popen('pwd').read()
        # # appNum = int(appNum[appNum.rfind("app")+3])
        # writeResult(appNum, endDecode, "decodingDelay")

        # startMerge = perf_counter_s()
        batch_results = Results()
        batch_results.combine_results(
            results, self.config.intersection_threshold)

        # need to merge this because all previous experiments assumed
        # that low (mpeg) results are already merged
        batch_results = merge_boxes_in_results(
            batch_results.regions_dict, 0.3, 0.3)

        batch_results.combine_results(
            rpn, self.config.intersection_threshold)
        # endMerge = perf_counter_s() - startMerge
        # writeResult(appNum, endMerge, "mergeDelay")

        # startSimulate = perf_counter_s()
        detections, regions_to_query = self.simulate_low_query(
            start_fid, end_fid, "server_temp", batch_results.regions_dict,
            False, self.config.rpn_enlarge_ratio, False)
        # endSimulate = perf_counter_s() - startSimulate
        # writeResult(appNum, endSimulate, "simulateDelay")

        self.last_requested_regions = regions_to_query
        # if profiling, then the curr_fid is not changing
        self.curr_fid = min(self.curr_fid + self.config.batch_size, self.nframes) if start_fid == self.curr_fid else self.curr_fid

        # Make dictionary to be sent back
        detections_list = []
        for r in detections.regions:
            detections_list.append(
                [r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])
        req_regions_list = []
        for r in regions_to_query.regions:
            req_regions_list.append(
                [r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])

        return {
            "results": detections_list,
            "req_regions": req_regions_list,
            "infer_delay": infer_delay,
            "end_decode": end_decode
        }

    def perform_high_query(self, file_data):
        low_images_direc = "server_temp"
        cropped_images_direc = "server_temp-cropped"

        with open(os.path.join(cropped_images_direc, "temp.mp4"), "wb") as f:
            f.write(file_data.read())

        results, infer_delay, end_decode = self.emulate_high_query(
            low_images_direc, low_images_direc, self.last_requested_regions)

        results_list = []
        for r in results.regions:
            results_list.append([r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])

        self.perform_server_cleanup()
        # Perform server side cleanup for the next batch
        # t = Thread(target=self.perform_server_cleanup)
        # t.start()
        # writeResult(appNum, results_list, "secondIterDebug")

        return {
            "results": results_list,
            "req_region": [],
            "infer_delay": infer_delay,
            "end_decode": end_decode
        }

def detectImage(fid, image, resolution, min_object_size, detector):
    temp_results = Results()
    temp_rpn_regions = Results()
    # detection_results, rpn_results, infer_temp = (
    #     detector.server_infer(image))
    frame_with_no_results = True
    for label, conf, (x, y, w, h) in detection_results:
        if (min_object_size and
                w * h < min_object_size) or w * h == 0.0:
            continue
        r = Region(fid, x, y, w, h, conf, label,
                    resolution, origin="mpeg")
        temp_results.append(r)
        frame_with_no_results = False
    for label, conf, (x, y, w, h) in rpn_results:
        r = Region(fid, x, y, w, h, conf, label,
                    resolution, origin="generic")
        temp_rpn_regions.append(r)
        frame_with_no_results = False
    # self.logger.debug(
    #     f"Got {len(temp_results)} results "
    #     f"and {len(rpn_regions)} for {fname}")

    if frame_with_no_results:
        temp_results.append(
            Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))
    return [temp_results, temp_rpn_regions]
