import pandas as pd
import os
import shutil
from dds_utils import (compress_and_get_size, crop_images, Results, compute_regions_size, 
                        Region, read_results_txt_dict, cleanup, read_results_dict, evaluate_partial, write_stats)
video = "lagrange"
ground_truth_dict = read_results_dict(f"./dds-adaptive/workspace/results/{video}_gt")
res = read_results_txt_dict(f"/tmp/ramdisk/VAP-Concierge/src/app/full_result_iter2/{video}_adaptive_520_0.0_twosides_batch_1_0.5_0.8_0.4")
tp, fp, fn, _, _, _, f1 = evaluate_partial(0,
                    90, res, ground_truth_dict,
                    0.3, 0.5, 0.4, 0.4, 0.5)

print(f1)