import os
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from torch import optim
from tools.load_data_from_author import re_load
from models.paragraph_level_BiLSTM import MyModel
from models.paragraph_level_BiLSTM import AuthorModel
from tools.load_data_from_json import get_paragraph_data
from tools.devide_train_batch import get_train_batch_list
# from train_valid_test.test_paragraph_level_model import test_fn
from train_valid_test.train_valid_paragraph_level_model import train_and_valid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--METHOD', type=int, default=0)
parser.add_argument('--FREEZE', type=int, default=0)
parser.add_argument('--EPOCHs', type=int, default=50)
parser.add_argument('--EX_LOSS', type=int, default=0)
parser.add_argument('--DROPOUT', type=float, default=0.5)
parser.add_argument('--BATCH_SIZE', type=int, default=128)
# parser.add_argument('--RANDOM_SEED', type=int, default=1234)  #
parser.add_argument('--LEARN_RATE', type=float, default=1e-3)
parser.add_argument('--IF_USE_EX_INITIAL_1', type=int, default=0)
parser.add_argument('--IF_USE_EX_INITIAL_2', type=int, default=0)
args = parser.parse_args()
print(args)

if __name__ == '__main__':

    valid_best_f1_list = []
    valid_best_acc_list = []

    dim = 343
    for t in range(1230, 1235):
        print("\ntime=", t)

        train_data_list, test_data_list = re_load(random_seed=t)  # from author
        # train_data_list, test_data_list = get_paragraph_data(dim=dim, random_seed=t)
        train_batch_list = get_train_batch_list(train_data_list, args.BATCH_SIZE, each_data_len=0)  #

        if args.METHOD == 0:
            model = MyModel(input_dim=dim,
                            dropout=args.DROPOUT,
                            if_use_ex_initial_1=args.IF_USE_EX_INITIAL_1,
                            if_use_ex_initial_2=args.IF_USE_EX_INITIAL_2,
                            random_seed=t,
                            ex_loss=args.EX_LOSS,
                            freeze=args.FREEZE).cuda()
        else:
            model = AuthorModel(input_dim=dim,
                                dropout=args.DROPOUT,
                                if_use_ex_initial_1=args.IF_USE_EX_INITIAL_1,
                                if_use_ex_initial_2=args.IF_USE_EX_INITIAL_2,
                                random_seed=t).cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.LEARN_RATE,
                               weight_decay=1e-4)

        best_epoch, best_model, best_macro_Fscore, best_acc = train_and_valid(model, optimizer, train_batch_list, test_data_list, args.EPOCHs)
        print("\nbest_epoch: ", best_epoch, best_macro_Fscore, best_acc)

        if args.METHOD == 0 and args.IF_USE_EX_INITIAL_1 == 0 and args.IF_USE_EX_INITIAL_2 == 0:
            best_model.save()

        valid_best_f1_list.append(best_macro_Fscore)
        valid_best_acc_list.append(best_acc)

    # ###############################
    print("valid f1: ", np.mean(np.array(valid_best_f1_list)), valid_best_f1_list)
    print("valid acc:", np.mean(np.array(valid_best_acc_list)), valid_best_acc_list)
