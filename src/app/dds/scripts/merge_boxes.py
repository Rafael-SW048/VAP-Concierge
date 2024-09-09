from concurrent.futures import ProcessPoolExecutor
import os

from app.dds.dds_utils import (
    Region,
    Results,
    merge_boxes_in_results as merge_boxes,
)

  
try: 
    NFRAMES = int(os.environ["NFRAMES"])
except KeyError or ValueError:
    NFRAMES = 300


def read_offline_result(result_path, start_fid, end_fid) -> Results:

    res: Results = Results()

    with open(result_path, "r") as result_fd:
        for line in result_fd:
            row = line.split(sep=",")
            fid = int(row[0])
            if start_fid <= fid < end_fid:
                res.add_single_result(Region(
                    fid, x=float(row[1]), y=float(row[2]),
                    w=float(row[3]), h=float(row[4]),
                    conf=float(row[6]), label=row[5],
                    resolution=float(row[7])
                    ))
    return res

def worker(filename, original_results_dir, merged_results_dir):
    if filename != "profile_bw" and filename != "profile_bw_frame":
        results = read_offline_result(
                os.path.join(original_results_dir, filename), 0, NFRAMES)
        merged_results = merge_boxes(results.regions_dict, 0.5, 0.8)
        merged_results.write_results_txt(
            os.path.join(merged_results_dir, filename))

def main():

    try:
        original_results_dir = os.environ["ORIGIN"]
        merged_results_dir = os.environ["MERGED"]
    except KeyError:
        return -1

    executor: ProcessPoolExecutor = ProcessPoolExecutor()

    all_files_name = os.listdir(original_results_dir)
    [executor.submit(worker, filename,
                    original_results_dir, merged_results_dir)
     for filename in all_files_name]


if __name__ == "__main__":
    main()
