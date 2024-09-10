"""
    entrance.py - user entrance for the platform
    author: Qizheng Zhang (qizhengz@uchicago.edu)
            Kuntai Du (kuntai@uchicago.edu)
"""

import os
import subprocess
import yaml
import sys
import signal
import logging

play_video_process = None

# dirty fix
sys.path.append('../')

# Set up logging
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("entrance.log")
file_handler.setLevel(logging.DEBUG)

# stream_handler = logging.StreamHandler(sys.stdout)  # Log to stdout for tee
# stream_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
# stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

def load_configuration():
    """read configuration information from yaml file

    Returns:
        dict: information of the yaml file
    """
    logger.info("Loading configuration file")
    with open('configuration.yml', 'r') as config:
        config_info = yaml.load(config, Loader=yaml.FullLoader)
    logger.debug("Configuration loaded: %s", config_info)
    return config_info

def execute_single(single_instance):
    """execute an atomic instance

    Args:
        single_instance (dict): the instance to be executed
    """
    # unpacking
    baseline = single_instance['method']
    logger.info("Executing single instance: %s", single_instance)

    # branching based on baselines
    if baseline == 'gt':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        result_file_name = f"{video_name}_gt"

        # skip if result file already exists
        if single_instance['overwrite'] == False and os.path.exists(os.path.join("results", result_file_name)):
            logger.info(f"Skipping {result_file_name} because it already exists.")
        else:
            single_instance['video_name'] = f'results/{result_file_name}'
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = 'stats'

            logger.debug("Running play_video.py with GT method")
            subprocess.run(['python', '../play_video.py', yaml.dump(single_instance)])

    elif baseline == 'mpeg':
        # unpacking
        video_name = single_instance['video_name']
        mpeg_qp = single_instance['low_qp']
        mpeg_resolution = single_instance['low_resolution']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        result_file_name = f"{video_name}_mpeg_{mpeg_resolution}_{mpeg_qp}"

        # skip if result file already exists
        if single_instance['overwrite'] == False and os.path.exists(os.path.join("results", result_file_name)):
            logger.info(f"Skipping {result_file_name} because it already exists.")
        else:
            single_instance['video_name'] = f'results/{result_file_name}'
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = 'stats'
            single_instance['ground_truth'] = f'results/{video_name}_gt'

            logger.debug("Running play_video.py with MPEG method")
            subprocess.run(['python', '../play_video.py', yaml.dump(single_instance)])

    elif baseline == 'dds':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        low_qp = single_instance['low_qp']
        high_qp = single_instance['high_qp']
        low_res = single_instance['low_resolution']
        high_res = single_instance['high_resolution']
        rpn_enlarge_ratio = single_instance['rpn_enlarge_ratio']
        batch_size = single_instance['batch_size']
        prune_score = single_instance['prune_score']
        objfilter_iou = single_instance['objfilter_iou']
        size_obj = single_instance['size_obj']
        adaptive_mode = single_instance['adaptive_mode']
        adaptive_test_display = single_instance['adaptive_test_display']

        # skip if result file already exists
        if adaptive_mode:
            result_file_name = (f"{video_name}_adaptive_{adaptive_test_display}_"
                                f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                                f"{prune_score}_{objfilter_iou}_{size_obj}")
        else:
            result_file_name = (f"{video_name}_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                                f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                                f"{prune_score}_{objfilter_iou}_{size_obj}")
        
        if single_instance['overwrite'] == False and os.path.exists(os.path.join("results", result_file_name)):
            logger.info(f"Skipping {result_file_name} because it already exists.")
        else:
            single_instance['real_video_name'] = video_name
            single_instance['video_name'] = f'results/{result_file_name}'
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = 'stats'
            single_instance['ground_truth'] = f'results/{video_name}_gt'
            single_instance['low_results_path'] = f'results/{video_name}_mpeg_{low_res}_{low_qp}'
            single_instance['profile_folder_path'] = f'{data_dir}/{video_name}'

            if single_instance["mode"] == 'implementation':
                assert single_instance['hname'] != False, "Must provide the server address for implementation, abort."
            
            logger.debug("Running play_video.py with DDS method")
            global play_video_process
            play_video_process = subprocess.Popen(['python', '../play_video.py', yaml.dump(single_instance)])

            play_video_process.wait()

def parameter_sweeping(instances, new_instance, keys):
    """recursive function for parameter sweeping

    Args:
        instances (dict): the instance in process
        new_instance (dict): recursive parameter
        keys (list): keys of the instance in process
    """
    if not keys: # base case
        execute_single(new_instance)
    else: # recursive step
        curr_key = keys[0]
        if isinstance(instances[curr_key], list):
            # need parameter sweeping
            for each_parameter in instances[curr_key]:
                # replace the list with a single value
                new_instance[curr_key] = each_parameter
                # proceed with the other parameters in keys
                parameter_sweeping(instances, new_instance, keys[1:])
        else: # no need for parameter sweeping
            new_instance[curr_key] = instances[curr_key]
            parameter_sweeping(instances, new_instance, keys[1:])

def execute_all(config_info):
    """execute all instances based on user's config info and default config info

    Args:
        config_info (dict): configuration information from the yaml file
    """
    all_instances = config_info['instances']
    default = config_info['default']
    logger.info("Executing all instances")

    for single_instance in all_instances:
        # propagate default config to current instance
        for key in default.keys():
            if key not in single_instance.keys():
                single_instance[key] = default[key]

        keys = list(single_instance.keys())
        new_instance = {} # initially empty
        parameter_sweeping(single_instance, new_instance, keys)

def clean_up(signum, frame):
    global play_video_process
    if play_video_process:
        logger.info("Terminating play_video_process")
        play_video_process.terminate()

if __name__ == "__main__":
    logger.info("Starting platform")
    
    # load configuration information (only once)
    config_info = load_configuration()
    data_dir = config_info['data_dir']
    
    signal.signal(signal.SIGINT, clean_up)

    try:
        execute_all(config_info)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
