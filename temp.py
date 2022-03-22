import os
import random
import sys
from tools.print_evaluation_result import print_evaluation_result

import torch
sys.path.append(os.getcwd() + '/resource')
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)

# from tools.statistic_file import statistic
# statistic()


def get_pre(main_verb_contribution_):
    main_verb_contribution = [contribution * 100000 for contribution in main_verb_contribution_]
    r = random.randint(0, 100000)

    if r > sum(main_verb_contribution[:6]):
        return 6
    if r > sum(main_verb_contribution[:5]):
        return 5
    if r > sum(main_verb_contribution[:4]):
        return 4
    if r > sum(main_verb_contribution[:3]):
        return 3
    if r > sum(main_verb_contribution[:2]):
        return 2
    if r > sum(main_verb_contribution[:1]):
        return 1

    return 0


test_dict_plus = torch.load('resource/statistic_dict_plus_test.pt')
train_dict_plus = torch.load('resource/statistic_dict_plus_train.pt')

pre_labels_list = []
gold_labels_list = []
for i in range(len(list(test_dict_plus.keys()))):
    main_verb = list(test_dict_plus.keys())[i]
    if test_dict_plus[main_verb][0][7] > 0:
        for j in range(7):
            gold_labels_list += [j for _ in range(test_dict_plus[main_verb][0][j])]

        if main_verb not in list(test_dict_plus.keys()):
            main_verb_contribution = test_dict_plus['None'][1]
        else:
            main_verb_contribution = test_dict_plus[main_verb][1]

        for _ in range(test_dict_plus[main_verb][0][7]):
            pre_labels_list.append(get_pre(main_verb_contribution))

print_evaluation_result(gold_labels_list, pre_labels_list)