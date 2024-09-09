import os
import numpy
import random


dataset=["aberdeen", "boston", "india", "hochiminh", "jakarta", "jakarta-uav", "lagrange", "miami", "roppongi", "tilton", "timesquare"]

combs = []

for i in range(len(dataset)-2):
    for j in range(i+1, len(dataset)-1):
        for k in range(j+1, len(dataset)):
            combs.append(f"- {dataset[i]} {dataset[j]} {dataset[k]}")
random.shuffle(dataset)
random.shuffle(dataset)
print(random.choices(combs, k=10))