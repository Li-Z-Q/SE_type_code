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

from tools.statistic_file import statistic
statistic()
test_dict_plus = torch.load('resource/statistic_dict_plus_test_replace.pt')
# train_dict_plus = torch.load('resource/statistic_dict_plus_train_replace.pt')
print(test_dict_plus)