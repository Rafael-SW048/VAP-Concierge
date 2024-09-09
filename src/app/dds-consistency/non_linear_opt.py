from dds_utils import get_best_configuration
bw_list = [200, 300, 400, 500, 600]
f1_score_10_boston = [0 for i in range(5)]
for bw in enumerate(bw_list):
    f1_temp = 0
    for i in range(10):
        _, _, _, _, f1_best, _ = get_best_configuration(0.45*bw[1], f'./data-set/uav-1/profile/profile-{i}.csv')
        f1_temp += f1_best
    f1_score_10_boston[bw[0]] = f1_temp/10


f1_score_10_jakarta = [0 for i in range(5)]
for bw in enumerate(bw_list):
    f1_temp = 0
    for i in range(10):
        _, _, _, _, f1_best, _ = get_best_configuration(0.45*bw[1], f'./data-set/jakarta/profile/profile-{i}.csv')
        f1_temp += f1_best
    f1_score_10_jakarta[bw[0]] = f1_temp/10


f1_score_10_highway = [0 for i in range(5)]
for bw in enumerate(bw_list):
    f1_temp = 0
    for i in range(10):
        _, _, _, _, f1_best, _ = get_best_configuration(0.45*bw[1], f'./data-set/roppongi/profile/profile-{i}.csv')
        f1_temp += f1_best
    f1_score_10_highway[bw[0]] = f1_temp/10
# print(f1_score_10_boston, f1_score_10_jakarta, f1_score_10_highway)


# now do the non linear optimization
# hardcoded as well
combs = [(400,400,400), (300,400,500), (200,400,600), (300,400,500), (600,400,200), (400,300,500), (400,200,600), (400,500,300), (400,600,200), (300, 500, 400), (200, 600, 400), (500, 300, 400), (600, 200, 400)]

f1_comb = []
for comb in combs:
    f1_comb.append(f1_score_10_boston[comb[0]//100 - 2]+f1_score_10_jakarta[comb[1]//100 -2]+f1_score_10_highway[comb[2]//100 -2])
print(f1_comb)
print(combs[f1_comb.index(max(f1_comb))])