"""
Using the result to calculate f1 score of a segment of the video.
Run from the workspace folder
"""

import sys

sys.path.append("../")

from dds_utils import evaluate_partial, read_results_dict

class calculator:

    def __init__(self) -> None:

        self.real_video_name = "rene"

        self.min_fid = 45
        self.max_fid = 49

        self.rpn_enlarge_ratio = 0.0
        self.low_threshold = 0.3

        self.app_path = ".."
        self.workspace_path = f"{self.app_path}/workspace"
        self.dataset_path = f"{self.app_path}/dataset/{self.real_video_name}"

    def segmental_f1(self, low_resoultion, high_resolution, low_qp, high_qp) -> float:
        results = read_results_dict(f"{self.dataset_path}/results/{self.real_video_name}_dds_{low_resoultion}_{high_resolution}_{low_qp}_{high_qp}_{self.rpn_enlarge_ratio}_twosides_batch_5_0.5_0.8_0.4")
        gt = read_results_dict(f"{self.workspace_path}/results/{self.real_video_name}_gt")

        _, _, _, _, _, _, f1 = evaluate_partial(
            min_fid = self.min_fid, 
            max_fid = self.max_fid, 
            map_dd = results,
            map_gt = gt,
            gt_confid_thresh = self.low_threshold, 
            mpeg_confid_thresh = 0.5, 
            max_area_thresh_gt = 0.4, 
            max_area_thresh_mpeg = 0.4)

        return f1
    
    def relative_f1(self, low_resolution_1, high_resolution_1, low_qp_1, high_qp_1, low_resolution_2, high_resolution_2, low_qp_2, high_qp_2) -> float:
        results_1 = read_results_dict(f"{self.dataset_path}/results/{self.real_video_name}_dds_{low_resolution_1}_{high_resolution_1}_{low_qp_1}_{high_qp_1}_{self.rpn_enlarge_ratio}_twosides_batch_5_0.5_0.8_0.4")
        results_2 = read_results_dict(f"{self.dataset_path}/results/{self.real_video_name}_dds_{low_resolution_2}_{high_resolution_2}_{low_qp_2}_{high_qp_2}_{self.rpn_enlarge_ratio}_twosides_batch_5_0.5_0.8_0.4")

        _, _, _, _, _, _, f1 = evaluate_partial(
            min_fid = self.min_fid, 
            max_fid = self.max_fid, 
            map_dd = results_1,
            map_gt = results_2,
            gt_confid_thresh = self.low_threshold, 
            mpeg_confid_thresh = 0.5, 
            max_area_thresh_gt = 0.4, 
            max_area_thresh_mpeg = 0.4)

        return f1

# profile test
print(calculator().segmental_f1(0.4, 1.0, 30, 28))

# estimiation test
print(1 - calculator().relative_f1(0.4, 1.0, 30, 28, 0.7, 0.8, 34, 28))