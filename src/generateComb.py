#!/bin/python
import numpy as np

# Do iteration for baseline first
AVAIL_DATASETS=["uav-1", "hochiminh", "jakarta", "jakarta-uav", "lagrange", "timesquare", "miami", "roppongi", "coldwater", "highway"]
IS_FAKEDs=["1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]

combination = []
combFaked = []
for i in range(len(AVAIL_DATASETS)-2):
    firstVid = AVAIL_DATASETS[i]
    firstFaked = IS_FAKEDs[i]
    for j in range(i+1,len(AVAIL_DATASETS)-1):
        secondVid = AVAIL_DATASETS[j]
        secondFaked = IS_FAKEDs[j]
        for k in range(j+1, len(AVAIL_DATASETS)):
            thirdVid = AVAIL_DATASETS[k]
            thirdFaked = IS_FAKEDs[k]
            combination.append(f"- {firstVid} {secondVid} {thirdVid}")
            combFaked.append(f"- {firstFaked} {secondFaked} {thirdFaked}")

combination = np.array(combination)
np.random.shuffle(combination)
f = open("listOfCombination.csv", "a")
for comb in combination:
    f.write(comb + '\n')
f.close()

f = open("listOfFaked.csv", "a")
for comb in combFaked:
    f.write(comb + '\n')
f.close()


