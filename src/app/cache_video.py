import re
import os
import csv
import sys
import yaml
import shutil
import subprocess
import numpy as np
import pandas as pd
import cv2 as cv
from typing import Tuple
from time import sleep, perf_counter as perf_counter_s
import tempfile
import mmap
from scipy import misc


# extract the params
# for now we only consider dds-adaptive app
appNum = int(sys.argv[1])
dataset = str(sys.argv[2])
mode = str(sys.argv[3])


tempDir = os.path.join(tempfile.gettempdir(), f"ramdisk/VAP-Concierge/src/app/app{appNum}/data-set/{dataset}/src/")
os.makedirs(tempDir, exist_ok=True)

# read image and then save to the temp directory
frames = os.listdir(f"/tmp/ramdisk/VAP-Concierge/src/app/{mode}/data-set/{dataset}/src/")
length = len([frame for frame in frames if ".png" in frame])

for i in range(length):
    image_path = f"/tmp/ramdisk/VAP-Concierge/src/app/{mode}/data-set/{dataset}/src/{str(i).zfill(10)}.png"
    image = cv.imread(image_path)
    tempFile = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=tempDir)
    np.save(tempFile, image)
    uid = 1000 # cc's uid
    gid = 1011
    os.chown(tempFile.name, uid, gid)
    os.rename(tempFile.name, f"{tempDir}{str(i).zfill(10)}.png")
    # change the ownership so it doesn't need to add sudo command

os.chown(tempDir, 1000, 1011)
os.chown(f"/tmp/ramdisk/VAP-Concierge/src/app/app{appNum}/data-set/{dataset}/", 1000, 1011)
