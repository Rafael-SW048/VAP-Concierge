import os
import re
import logging
import sys
import yaml
from munch import *
from backend.server import Server
from frontend.client import Client
from dds_utils import (
    ServerConfig,
    read_results_dict,
    evaluate,
    write_stats,
    read_bandwidth_limit,
    evaluate_partial,
    read_profile_info,
    writeResult
)
import signal

def main(args):
    # Configure logging based on verbosity level
    numeric_level = getattr(logging, args.verbosity.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.verbosity}')

    app_idx = os.environ["APP_IDX"]

    # Set up logging for this script
    logger = logging.getLogger(f"dds{app_idx}")
    numeric_level = logging.DEBUG  # Assume numeric_level is set earlier
    logger.setLevel(numeric_level)

    file_handler = logging.FileHandler(f"play_video_{app_idx}.log")
    file_handler.setLevel(numeric_level)

    # stream_handler = logging.StreamHandler(sys.stdout)  # Log to stdout for tee
    # stream_handler.setLevel(numeric_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    logger.info(f"Starting DDS server execution with high treshold of "
                f"{args.high_threshold} low threshold of "
                f"{args.low_threshold} tracker length of "
                f"{args.tracker_length}")
    logger.debug(f"Application Index: {app_idx}")
    logger.debug(f"Arguments received: {args}")

    # Initialize variables
    config = args
    server = None
    mode = None
    results, bw = None, None
    bandwidth_limit_dict = None

    try:
        if args.simulate:
            logger.error("Simulation mode is not supported anymore")
            raise NotImplementedError("Simulation mode is not supported anymore")
        elif not args.simulate and not args.hname and args.high_resolution != -1:
            mode = "emulation"
            logger.info(f"Running DDS in EMULATION mode on {args.video_name}")

            if args.adaptive_mode:
                mode = "emulation-adaptive"
                bandwidth_limit_path = os.path.join(
                    args.profile_folder_path,
                    args.profile_folder_name,
                    'bandwidthLimit.yml'
                )
                logger.debug(f"Reading bandwidth limit from {bandwidth_limit_path}")
                bandwidth_limit_dict = read_bandwidth_limit(bandwidth_limit_path)
                if len(bandwidth_limit_dict['frame_id']) > 1:
                    mode = "emulation-adaptive-separated"
                logger.debug(f"Adaptive mode set to {mode}")

            logger.info("Initializing server")
            server = Server(config)

            logger.info("Initializing client")
            client = Client(args.hname, config, server)

            logger.info("Starting video analysis in emulation mode")
            results, bw = client.analyze_video_emulate(
                args.video_name,
                args.high_images_path,
                args.enforce_iframes,
                args.low_results_path,
                args.debug_mode,
                args.adaptive_mode,
                bandwidth_limit_dict
            )
        elif not args.simulate and not args.hname:
            mode = "mpeg"
            logger.info(f"Running in MPEG mode with resolution {args.low_resolution} on {args.video_name}")

            logger.info("Initializing server")
            server = Server(config)

            logger.info("Initializing client")
            client = Client(args.hname, config, server)

            logger.info("Starting video analysis in MPEG mode")
            results, bw = client.analyze_video_mpeg(
                args.video_name,
                args.high_images_path,
                args.enforce_iframes
            )
        elif not args.simulate and args.hname:
            mode = "implementation"
            logger.info(f"Running DDS in IMPLEMENTATION mode with server at {args.hname} using video {args.video_name}")

            app_idx = os.environ["APP_IDX"]
            client_ip = os.environ["CLIENT_IP"]
            public_ip = os.environ["PUBLIC_IP"]
            ctrl_port = int(os.environ["CTRL_PRT"])
            is_faked = bool(int(os.environ["IS_FAKED"]))
            # uri = os.environ["PUBLIC_IP"]

            logger.debug(f"Environment variables - APP_IDX: {app_idx}, CLIENT_IP: {client_ip}, PUBLIC_IP: {public_ip}, CTRL_PRT: {ctrl_port}, IS_FAKED: {is_faked}")

            logger.info("Initializing client")
            client = Client(
                args.hname,
                config,
                server,
                app_idx=int(app_idx),
                uri=client_ip,
                edge_uri=public_ip,
                control_port=ctrl_port,
                is_faked=is_faked
            )

            profile_info_path = os.path.join(
                args.profile_folder_path,
                args.profile_folder_name,
                'profile_info.yml'
            )
            logger.debug(f"Reading profile info from {profile_info_path}")
            profile_info = read_profile_info(profile_info_path)

            logger.info("Starting video analysis in implementation mode")
            results, bw = client.analyze_video(
                args.video_name,
                args.high_images_path,
                config,
                args.enforce_iframes,
                profile_info
            )
        else:
            logger.error("Invalid configuration: Unable to determine execution mode")
            raise ValueError("Invalid configuration: Unable to determine execution mode")

        # Evaluation and writing results
        low_bw, high_bw = bw
        logger.debug(f"Bandwidth usage - Low: {low_bw}, High: {high_bw}")
        f1_score = 0
        stats = (0, 0, 0)
        number_of_frames = len(
            [x for x in os.listdir(args.high_images_path) if x.endswith(".png")]
        )
        completed_frames = client.completed_frames
        logger.debug(f"Number of frames: {number_of_frames}, Completed frames: {completed_frames}")

        if args.ground_truth:
            logger.info("Evaluating results against ground truth")
            ground_truth_dict = read_results_dict(args.ground_truth)
            logger.debug("Ground truth data loaded successfully")

            tp, fp, fn, _, _, _, f1_score = evaluate_partial(
                10,
                completed_frames - 1,
                results.regions_dict,
                ground_truth_dict,
                args.low_threshold,
                0.5,
                0.4,
                0.4
            )
            stats = (tp, fp, fn)
            logger.info(f"Evaluation complete - F1 Score: {f1_score}, TP: {tp}, FP: {fp}, FN: {fn}")
        else:
            logger.warning("No ground truth provided; skipping evaluation")

        # Writing F1 score to file
        appNum = os.popen('pwd').read()
        appNum = int(appNum[appNum.rfind("app")+3])
        f1_score_file = f"../../f1Score-{appNum}.csv"
        logger.debug(f"Writing F1 score to {f1_score_file}...")
        with open(f1_score_file, "a") as f:
            f.write(f"{f1_score}\n")
        logger.info(f"F1 score {f1_score} written to {f1_score_file}")

        # Write evaluation results to file
        output_bandwidth = bandwidth_limit_dict['bandwidth_limit'][0] if bandwidth_limit_dict else -1
        logger.debug(f"Writing stats to {args.outfile}")
        write_stats(
            args.outfile,
            args.video_name,
            config,
            f1_score,
            stats,
            bw,
            completed_frames,
            output_bandwidth,
            mode
        )
        logger.info("Statistics written successfully")

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)

def signal_handler(signal_received, frame):
    logger = logging.getLogger("dds_signal_handler")
    logger.info(f"Signal {signal_received} received. Terminating process gracefully.")
    sys.exit(0)

if __name__ == "__main__":
    # Parse arguments from command line
    try:
        args = munchify(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))
        logger = logging.getLogger("dds_main")

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.debug("Performing argument validations")

        # Argument validations
        if not args.simulate and not args.hname and args.high_resolution != -1:
            if not args.high_images_path:
                logger.error("High images path is required for DDS emulation mode")
                sys.exit(1)

        if not re.match(r"DEBUG|INFO|WARNING|ERROR|CRITICAL", args.verbosity.upper()):
            logger.error("Invalid verbosity level provided. Valid options are:\n"
                         "\tdebug\n\tinfo\n\twarning\n\terror")
            sys.exit(1)

        if args.estimate_banwidth and not args.high_images_path:
            logger.error("High images path is required to estimate bandwidth")
            sys.exit(1)

        if not args.simulate and args.high_resolution != -1:
            if args.low_images_path:
                logger.warning("Low images path provided but will be ignored")
                args.low_images_path = None
            args.intersection_threshold = 1.0

        if args.method != "dds":
            assert args.high_resolution == -1, "Only DDS method supports two quality levels"

        if args.high_resolution == -1:
            logger.info("Only one resolution provided; running MPEG emulation")
            assert args.high_qp == -1, "MPEG emulation supports only one QP value"
        else:
            assert args.low_resolution <= args.high_resolution, \
                "Low resolution({args.low_resolution}) cannot be greater than high resolution({args.high_resolution})"
            assert not (args.low_resolution == args.high_resolution and args.low_qp < args.high_qp), \
                "For the same resolution, low QP({args.low_qp}) cannot be greater than high QP({args.high_qp})"

        logger.info("Starting main execution")
        main(args)
    except Exception as e:
        logger = logging.getLogger("dds_init")
        logger.exception(f"Failed to initialize application: {e}")
        sys.exit(1)
