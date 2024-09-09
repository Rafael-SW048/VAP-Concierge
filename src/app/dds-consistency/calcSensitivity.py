from dds_utils import get_best_configuration
from statistics import mean, median

dataset=["uav-1","hochiminh","jakarta","jakarta-uav","lagrange","timesquare","miami","roppongi","coldwater","highway"]

# generate permutations
perms = []
for i in range(len(dataset)-1):
    for k in range(i+1,len(dataset)):
        perms.append([dataset[i], dataset[k]])
        perms.append([dataset[k], dataset[i]])

# delta_bw = [16, 80]
# main_bw = [400, 400]
delta_bw = [20, 80]
main_bw = [[300, 300], [350, 350], [400, 400], [450, 450]]
budget = [0.45, 1]
segments = [100, 20]
directory = ["", "-5"]
# for k in range(len(main_bw)):
#     for j in range(len(main_bw[k])):
#         sens_dict = {"5":[], "1":[]}
#         count = 0
#         sum = 0
#         for i in range(segments[j]):
#             for perm in perms:
#                 _,_,_,_,f1_curr,_ = get_best_configuration(main_bw[k][j]*budget[j], f"./data-set/{perm[0]}/profile{directory[j]}/profile-{i}.csv")
#                 _,_,_,_,f1_high,_ = get_best_configuration(main_bw[k][j]*budget[j] + delta_bw[j], f"./data-set/{perm[0]}/profile{directory[j]}/profile-{i}.csv")
#                 diff_high = f1_high - f1_curr
#                 _,_,_,_,f1_curr,_ = get_best_configuration(main_bw[k][j]*budget[j], f"./data-set/{perm[1]}/profile{directory[j]}/profile-{i}.csv")
#                 _,_,_,_,f1_low,_ = get_best_configuration(main_bw[k][j]*budget[j] - delta_bw[j], f"./data-set/{perm[1]}/profile{directory[j]}/profile-{i}.csv")
#                 diff_low = f1_curr - f1_low
#                 sensitivity = diff_high - diff_low
#                 if  sensitivity > 0:
#                     sum += sensitivity
#                     count += 1
#         print(sum/count)

# sensitivities = []
# for bw in main_bw:
#     totSens = []
#     print(bw[0])
#     firstVal = 0
#     for video in dataset:
#         f1_scores = []
#         for i in range(100):
#             _,_,_,_,f1_curr,_ = get_best_configuration(bw[0]*0.45, f"./data-set/{video}/profile/profile-{i}.csv")
#             if len(f1_scores) != 0:
#                 f1_scores.append(abs(f1_scores[-1] - f1_curr))
#             else:
#                 f1_scores.append(f1_curr)
#         totSens.append(median(f1_scores[1::]))
#     sensitivities.append(totSens)
# print(sensitivities)

# sensitivities = []
# for bw in main_bw:
#     totSens = []
#     firstVal = 0
#     for video in dataset:
#         f1_scores = []
#         for i in range(20):
#             _,_,_,_,f1_curr,_ = get_best_configuration(bw[0], f"./data-set/{video}/profile-5/profile-{i}.csv")
#             if len(f1_scores) != 0:
#                 f1_scores.append(abs(f1_scores[-1] - f1_curr))
#             else:
#                 f1_scores.append(f1_curr)
#         totSens.append(median(f1_scores[1::]))
#     sensitivities.append(totSens)
# print(sensitivities)


delta_bw = 20
main_bw = [300+i*50 for i in range(4)]
budget = 0.45
segments = 100
# directory = ["", "-5"]

for bw in main_bw:
    sensitivity = []
    for i in range(segments):
        _,_,_,_,f1_high,_ = get_best_configuration(budget*bw+delta_bw, f"./data-set/uav-1/profile/profile-{i}.csv")
        _,_,_,_,f1_low,_ = get_best_configuration(budget*bw-delta_bw, f"./data-set/uav-1/profile/profile-{i}.csv")
        sens = (f1_high - f1_low)/(2*delta_bw)
        sensitivity.append(sens)
    print(sensitivity)