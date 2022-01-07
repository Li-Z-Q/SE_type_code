import os
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch import optim
from tools.get_paragraph_level_data import get_data
from tools.devide_train_batch import get_train_batch_list
from train_valid_test.test_paragraph_level_model import test_model
from models.paragraph_level_BiLSTM_try_sim_weight_average_gold import MyModel
from train_valid_test.train_valid_paragraph_level_model import train_and_valid


EPOCHs = 40
DROPOUT = 0.5
BATCH_SIZE = 128
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-4

if __name__ == '__main__':

    test_f1_list = []
    test_acc_list = []
    valid_best_f1_list = []
    valid_best_acc_list = []

    for t in range(5):
        print("\n\n\n\ntime=", t)

        train_data_list, valid_data_list, test_data_list = get_data(if_do_embedding=True, stanford_path='stanford-corenlp-4.3.1')
        train_batch_list = get_train_batch_list(train_data_list, BATCH_SIZE, each_data_len=0)

        model = MyModel(dropout=DROPOUT).cuda()
        optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

        best_epoch, best_model, best_macro_Fscore, best_acc = train_and_valid(model, optimizer, train_batch_list, valid_data_list, EPOCHs)
        torch.save(best_model, 'output/model_paragraph_level_BiLSTM_try_sim_weight_average_gold.pt')
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
