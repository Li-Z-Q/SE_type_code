import os
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)

from tools.load_data_from_author import re_load

re_load(0)