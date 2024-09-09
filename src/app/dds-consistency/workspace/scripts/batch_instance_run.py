import pandas as pd
import subprocess

# base_profile_path = 'results_archive/01-02-2-rene/profile.csv'
target_configuration_path = 'configuration.yml'

profile_folder_name = 'profile'
target_bandwidth_limit_path = '../dataset/rene/profile/bandwidthLimit.yml'

profile_folder_name_separated = 'profile-separated'
target_bandwidth_limit_path_separated = '../dataset/rene/profile-separated/bandwidthLimit.yml'

# profile = pd.read_csv(base_profile_path)

for bandwidth in range(500, 10500 + 500, 500):
    subprocess.run(f"yq -yi '.default.adaptive_test_display = \"{bandwidth}\"' {target_configuration_path}", shell = True)
    subprocess.run(f"yq -yi '.default.profile_folder_name = \"{profile_folder_name}\"' {target_configuration_path}", shell = True)
    subprocess.run(f"yq -yi '.bandwidth_limit = {[bandwidth]}' {target_bandwidth_limit_path}", shell = True)
    subprocess.run("python entrance.py", shell = True)
    subprocess.run(f"yq -yi '.default.adaptive_test_display = \"SEPARATED-{bandwidth}\"' {target_configuration_path}", shell = True)
    subprocess.run(f"yq -yi '.default.profile_folder_name = \"{profile_folder_name_separated}\"' {target_configuration_path}", shell = True)
    subprocess.run(f"yq -yi '.bandwidth_limit = {[bandwidth]*4}' {target_bandwidth_limit_path_separated}", shell = True)
    subprocess.run("python entrance.py", shell = True)