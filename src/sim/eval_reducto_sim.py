
def check_accuracy(infer_path, gt_path):
    frame_inferred = 0
    with open(infer_path, "r") as infer_fd:
        for line in infer_fd:
            frame_inferred += int(line)

    num_trigger_frame = 0
    with open(gt_path, "r") as gt_fd:
        for line in gt_fd:
            fid = int(line)
            if fid <= 3600:
                num_trigger_frame += 1
            else:
                break

    return frame_inferred / num_trigger_frame


def print_gt(gt_path, output):
    with open(output, "w+") as out_fd:
        with open(gt_path, "r") as gt_fd:
            seg_id = 0
            while seg_id < 120:
                fid_min = 30 * seg_id
                fid_max = fid_min + 30
                inferred_frame = 0
                for line in gt_fd:
                    if fid_min < int(line) < fid_max:
                        inferred_frame += 1
                seg_id += 1
                out_fd.write(str(inferred_frame) + "\n")
                gt_fd.seek(0)


def print_baseline(gt_path, output, fps):
    with open(output, "w+") as out_fd:
        with open(gt_path, "r") as gt_fd:
            seg_id = 0
            while seg_id < 120:
                fid_min = 30 * seg_id
                fid_max = fid_min + 30
                inferred_frame = 0
                for line in gt_fd:
                    if fid_min < int(line) < fid_max and inferred_frame < fps:
                        inferred_frame += 1
                seg_id += 1
                out_fd.write(str(inferred_frame) + "\n")
                gt_fd.seek(0)


def check_baseline(gt_path, fps):

    detections = 0
    with open(gt_path, "r") as gt_fd:
        seg_id = 0
        while seg_id < 120:
            fid_min = 30 * seg_id
            fid_max = fid_min + 30
            inferred_frame = 0
            for line in gt_fd:
                if fid_min < int(line) < fid_max\
                        and inferred_frame < fps:
                    inferred_frame += 1
            detections += inferred_frame
            seg_id += 1
            gt_fd.seek(0)
        print(str(detections))

    num_trigger_frame = 0
    with open(gt_path, "r") as gt_fd:
        for line in gt_fd:
            fid = int(line)
            if fid <= 3600:
                num_trigger_frame += 1
            else:
                break
    return detections / num_trigger_frame


if __name__ == "__main__":
    print_gt("sim/reducto-mix1", "sim/reducto1-gt")
    print_gt("sim/reducto-mix2", "sim/reducto2-gt")
    print_baseline("sim/reducto-mix1", "sim/reducto1-baseline", 5)
    print_baseline("sim/reducto-mix2", "sim/reducto2-baseline", 5)
    # print(check_accuracy("sim/reducto1", "sim/reducto-mix1"))
    # print(check_baseline("sim/reducto-mix1", 5))
    # print(check_accuracy("sim/reducto2", "sim/reducto-mix2"))
    # print(check_baseline("sim/reducto-mix2", 5))
