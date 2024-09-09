import pandas as pd
import numpy as np
import os
from dds_utils import writeResult
import shutil
import json
import ast
from scipy import optimize as op
from processData import get_combination

length = 100

def get_data(run, mode):
    data_dict = {}
    for i in range(length):
        dict_temp = {}
        for j in range(3):
            data = None
            with open(f"./heatmap-dds_{run}-2/{mode}-{j+1}.csv", "r") as file:
            # with open(f"./{mode}-{j+1}.csv", "r") as file:
                iter = 0
                for line in file:
                    if iter == i:
                        data = ast.literal_eval(line)
                        # dict_temp[j] = data
                    iter += 1
            test = None
            with open(f"./heatmap-dds_{run}-2/isMinBW-{j+1}.csv", "r") as file:
                iter = 0
                for line in file:
                    if iter == i:
                        test = ast.literal_eval(line)
                    iter += 1
            if data:
                for k in range(len(data)):
                    data[k] += (test[k],)
                dict_temp[j] = data
            else:
                return data_dict
        data_dict[i] = dict_temp
    return data_dict

def normalize(arr):
    max = np.amax(arr)
    min = np.amin(arr)
    if max == 0:
        return arr
    if max - min == 0:
        return arr / max
    return (arr - min) / (max - min)

def concierge(arr, mode, frame):
    # arr = [data[0] for data in arr] # Upwards Direction
    # Both directions
    # arr = normalize(arr)
    arr_temp = []
    victim_loc = -1
    victim_infer_delta = 2
    is_min_bw = []
    for i in range(len(arr)):
        is_min_bw.append(arr[i][2])
        lowTotal = 0
        for j in range(len(arr)):
            if i!=j:
                lowTotal += arr[j][1]
        arr_temp.append(arr[i][0] - lowTotal)
        infer_delta = arr[i][0] + arr[i][1]
        if infer_delta < victim_infer_delta and not is_min_bw[i]:
            victim_infer_delta = infer_delta
            victim_loc = i
    if victim_loc == -1:
        return (-1,-1)
    for i in range(len(arr)):
        if i != victim_loc:
            arr_temp[i] += arr[victim_loc][1]
    # print(arr)
    if all([data <= 0 for data in arr_temp]):
        return (-1,-1)
    # if mode == "inferDiff":
    arr = normalize(arr_temp)
    normalized_infer_gradients = np.array(
            [data for data in arr])
    # res: List[Allocation] = []
    obj = normalized_infer_gradients * -1
    conservation_lhs: np.ndarray = np.array(
                [[1 for data in arr]])
    conservation_rhs: np.ndarray = np.array([0])
    boundaries: np.ndarray = np.array(
            # [(max(-request.app.current_bw * PROFILING_DELTA,
            [(0, 80) if i != victim_loc else (-80,0) 
             for i in range(len(arr))])
    lp_res: op.OptimizeResult = op.linprog(obj, A_eq=conservation_lhs,
                                b_eq=conservation_rhs, bounds=boundaries)
    # writeResult(1, f"frame: {frame}, mode: {mode}, sens: {arr}, result: {lp_res.x}", "conciergeResult")

    lp_res.x = np.round(lp_res.x, decimals=2)
    # print(lp_res.x)
    maxVal, minVal = max(lp_res.x), min(lp_res.x)
    maxApp, minApp = -1, -1
    if maxVal > 1 and minVal < -1:
        maxApp = np.where(lp_res.x == maxVal)[0][0]
        minApp = np.where(lp_res.x == minVal)[0][0]
        unique, counts = np.unique(lp_res.x, return_counts=True)
        counter = dict(zip(unique, counts))
        if counter[minVal] > 1:
            minApp = 3
    return (maxApp, minApp)


def simulate(inferDiff, accSen):
    minLength = min([len(inferDiff[i]) for i in range(3)])
    test = minLength
    # print(minLength)
    count = 0
    for i in range(minLength):
        logInferDiff = [inferDiff[j][i] for j in range(3)]
        logAccSen = [accSen[j][i] for j in range(3)]
        inferDiffDecision = concierge(logInferDiff, "inferDiff", i)
        accSenDecision = concierge(logAccSen, "accSen", i)
        writeResult(1, f"\n", "conciergeResult")
        # if i == 23:
        #     print(logInferDiff)
        #     print(logAccSen)
        # print()
        if inferDiffDecision == accSenDecision:
            # count += 1
            if inferDiffDecision != (-1,-1):
                # writeResult(1, f"frame: {i}, inferDiff-decision: {inferDiffDecision}, accSen-decision: {accSenDecision}", "decisionDebugger")
                count += 1
            else:
                # writeResult(1, f"frame: {i}, inferDiff-decision: {inferDiffDecision}, accSen-decision: {accSenDecision}", "decisionDebugger")
                test -= 1
        elif inferDiffDecision == (-1,-1):
            test -= 1
            # writeResult(1, f"frame: {i}, inferDiff-decision: {inferDiffDecision}, accSen-decision: {accSenDecision}", "decisionDebugger")
            pass
        # writeResult(1, f"frame: {i}, inferDiff-decision: {inferDiffDecision}, accSen-decision: {accSenDecision}", "decisionDebugger")
    # print(count)
    # print(test)
    return (count,test)
    


def main():
    # Get Data
    # inferDiff = get_data("inferDiff","inferDiff")
    inferDiff = get_data("accSen","inferDiff")
    accSen = get_data("accSen","accSen")
    combinations_dds_accSen = get_combination("heatmap-dds_accSen-2", "sanityCheck")
    consistencies = []
    longest = max([key+1 for key in inferDiff.keys()])
    print(longest)
    print(len(combinations_dds_accSen))
    for i in range(longest):
        consistency = simulate(inferDiff[i], accSen[i])
        consistencies.append(consistency)
    dict_1 = {}
    iter = 0
    for key in combinations_dds_accSen:
        if key not in dict_1.keys():
            dict_1[key] = consistencies[iter]
        else:
            try:
                dict_1[key] = consistencies[iter] if consistencies[iter][0]/consistencies[iter][1] > dict_1[key][0]/dict_1[key][1] else dict_1[key]
            except ZeroDivisionError:
                dict_1[key] = (0,0)
        iter += 1
    inferDiff = get_data("inferDiff","inferDiff")
    accSen = get_data("inferDiff","accSen")
    combinations_dds_inferDiff = get_combination("heatmap-dds_inferDiff-2", "sanityCheck")
    longest = max([key+1 for key in inferDiff.keys()])
    consistencies2 = []
    for i in range(longest):
        consistency = simulate(inferDiff[i], accSen[i])
        consistencies2.append(consistency)
    dict_2 = {}
    iter = 0
    for key in combinations_dds_inferDiff:
        if key not in dict_2.keys():
            dict_2[key] = consistencies2[iter]
        else:
            try:
                dict_2[key] = consistencies2[iter] if consistencies2[iter][0]/consistencies2[iter][1] > dict_2[key][0]/dict_2[key][1] else dict_2[key]
            except ZeroDivisionError:
                dict_2[key] = (0,0)
        iter += 1
    # print(dict_1['consistency(900, 30, jakarta miami coldwater)'][0] + dict_2['consistency(900, 30, jakarta miami coldwater)'][0])
    print(dict_1['consistency(1200, 30, boston jakarta lagrange)'])
    print(dict_2['consistency(1200, 30, boston jakarta lagrange)'])
    # cons_total = {}
    # for key in dict_1.keys():
    #     try:
    #         test = (dict_1[key][0]+dict_2[key][0])/(dict_1[key][1]+dict_2[key][1])
    #     except ZeroDivisionError:
    #         test = -1
    #     cons_total[key] = test
    # print(cons_total)
    # for key, val in cons_total.items():
    #     writeResult(1, key, "consistency_key")
    #     writeResult(1, val, "consistency_val")
    # writeResult(1, consistencies, "upwardsConsistency")
    # writeResult(1, consistencies, "bothDirectionsConsistency")


if __name__=="__main__":
    main()
