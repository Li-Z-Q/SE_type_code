import os
import sys

sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/resource')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)

import warnings

warnings.filterwarnings('ignore')

import torch
import numpy as np
import run_sentence_level_BiLSTM_extra
from torch import optim
from tools.get_paragraph_level_data import get_data
from tools.devide_train_batch import get_train_batch_list
from train_valid_test.test_paragraph_level_model import test_model
from models.paragraph_level_BiLSTM_label_embedding_MLP_pre import MyModel
from train_valid_test.train_valid_paragraph_level_model import train_and_valid

import argparse

parser = argparse.ArgumentParser(description='para transfer')
parser.add_argument('--EPOCHs', type=int, default=20)
parser.add_argument('--DROPOUT', type=float, default=0.5)
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--LEARN_RATE', type=float, default=1e-3)
parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-4)
parser.add_argument('--fold_num', type=int, default=0)
args = parser.parse_args()
print(args)

EPOCHs = args.EPOCHs
DROPOUT = args.DROPOUT
BATCH_SIZE = args.BATCH_SIZE
LEARN_RATE = args.LEARN_RATE
WEIGHT_DECAY = args.WEIGHT_DECAY
fold_num = args.fold_num

if __name__ == '__main__':

    test_f1_list = []
    test_acc_list = []
    valid_best_f1_list = []
    valid_best_acc_list = []

    for t in range(1):
        print("\n\n\n\ntime=", t)

        model = MyModel(dropout=DROPOUT, stanford_path='stanford-corenlp-4.3.1', pre_model_path='models/model_sentence_level_BiLSTM_extra.pt').cuda()
        optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

        train_data_list, valid_data_list, test_data_list = get_data(if_do_embedding=True,
                                                                    stanford_path='stanford-corenlp-4.3.1',
                                                                    random_seed=fold_num)
        train_batch_list = get_train_batch_list(train_data_list, BATCH_SIZE, each_data_len=0)

        run_sentence_level_BiLSTM_extra.main(train_data_list, valid_data_list, test_data_list)

        best_epoch, best_model, best_macro_Fscore, best_acc = train_and_valid(model, optimizer, train_batch_list,
                                                                              valid_data_list, EPOCHs)
        torch.save(best_model, 'output/model_paragraph_level_BiLSTM_label_embedding_MLP_pre.pt')
        print("best_epoch: ", best_epoch, best_macro_Fscore, best_acc)

        f1_score, acc = test_model(test_data_list, best_model)

        test_f1_list.append(f1_score)
        test_acc_list.append(acc)
        valid_best_f1_list.append(best_macro_Fscore)
        valid_best_acc_list.append(best_acc)

    # ###############################
    print("test f1:  ", np.mean(np.array(test_f1_list)), test_f1_list)
    print("test ass: ", np.mean(np.array(test_acc_list)), test_acc_list)
    print("valid f1: ", np.mean(np.array(valid_best_f1_list)), valid_best_f1_list)
    print("valid acc:", np.mean(np.array(valid_best_acc_list)), valid_best_acc_list)

    open('output/paragraph_level_BiLSTM/label_embedding_MLP_pre/test_acc/' + str(np.mean(np.array(test_acc_list))) + '.txt', 'w')
    open('output/paragraph_level_BiLSTM/label_embedding_MLP_pre/test_f1/' + str(np.mean(np.array(test_f1_list))) + '.txt', 'w')
    open('output/paragraph_level_BiLSTM/label_embedding_MLP_pre/valid_acc/' + str(np.mean(np.array(valid_best_acc_list))) + '.txt', 'w')
    open('output/paragraph_level_BiLSTM/label_embedding_MLP_pre/valid_f1/' + str(np.mean(np.array(valid_best_f1_list))) + '.txt', 'w')
