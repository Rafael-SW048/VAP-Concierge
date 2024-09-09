import shutil
import os

dataset = "rene"
segmentLength = 25

number_of_frames = len([f for f in os.listdir(dataset + "/src/") if ".png" in f])

segmentID = 0
for i in range(0, number_of_frames, segmentLength):
    os.mkdir(f"{dataset}-{segmentLength}-{segmentID}")
    os.mkdir(f"{dataset}-{segmentLength}-{segmentID}/src")
    for j in range(0, segmentLength):
        if (i+j < number_of_frames):
            src = f"{dataset}/src/{i+j:010}.png"
            des = f"{dataset}-{segmentLength}-{segmentID}/src/{j:010}.png"
            shutil.copy(src, des)
    segmentID += 1