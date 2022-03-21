# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import os
import sys
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from torch import optim
from tools.load_data_from_author import re_load
from models.sentence_level_Linear_3_class import MyModel
from tools.devide_train_batch import get_train_batch_list
from tools.load_data_from_json import from_paragraph_to_sentence
from train_valid_test.train_valid_sentence_level_model_3_class import train_and_valid_fn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--EPOCHs', type=int, default=30)
# parser.add_argument('--RANDOM_SEED', type=int, default=0)  #
parser.add_argument('--DROPOUT', type=float, default=0.5)
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--LEARN_RATE', type=float, default=1e-3)
parser.add_argument('--IF_USE_EX_INITIAL', type=int, default=0)
args = parser.parse_args()
print(args)

if __name__ == '__main__':

    valid_best_f1_list = []
    valid_best_acc_list = []

    dim = 343
    for t in range(1230, 1233):
        print("\n\ntime=", t)

        train_data_list, test_data_list = re_load(random_seed=t)
        test_data_list = from_paragraph_to_sentence(test_data_list, random_seed=t)
        train_data_list = from_paragraph_to_sentence(train_data_list, random_seed=t)
        train_batch_list = get_train_batch_list(train_data_list, args.BATCH_SIZE, each_data_len=1)

        model = MyModel(input_dim=dim,
                        dropout=args.DROPOUT,
                        random_seed=t,
                        if_use_ex_initial=args.IF_USE_EX_INITIAL).cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.LEARN_RATE,
                               weight_decay=1e-4)

        best_epoch, best_model, best_macro_Fscore, best_acc = train_and_valid_fn(model, optimizer, train_batch_list, test_data_list, args.EPOCHs)
        if args.IF_USE_EX_INITIAL == 0:
            best_model.save()
            print('3 Linear class sentence save')
        print("\ntime={}, best_epoch: ".format(t), best_epoch, best_macro_Fscore, best_acc)

        valid_best_acc_list.append(best_acc)
        valid_best_f1_list.append(best_macro_Fscore)

    print("\n\nvalid f1: ", np.mean(np.array(valid_best_f1_list)), valid_best_f1_list)
    print("valid acc:", np.mean(np.array(valid_best_acc_list)), valid_best_acc_list)