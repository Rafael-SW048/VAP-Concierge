import shutil
import pandas as pd

dataset_folder = "../dataset"

dataset = "uav-1"
profile_num = 20

profile_folder = f"{dataset_folder}/{dataset}/profile"

for i in range(profile_num):
    profiles = pd.read_csv(f"{profile_folder}/profile-{i}.csv")
    profiles["bandwidth"] = profiles["bandwidth"] + 250
    profiles["F1"] = profiles["F1"].round(3)

    profiles.to_csv(f"{profile_folder}/profile-{i}.csv", index=False)