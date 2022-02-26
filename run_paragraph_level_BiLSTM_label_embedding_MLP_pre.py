import os
import random
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
import run_sentence_level_BiLSTM_ex
from torch import optim
from tools.get_memory_house import get_memory
from tools.get_paragraph_level_data import get_data
from tools.devide_train_batch import get_train_batch_list
from train_valid_test.test_paragraph_level_model import test_model
from models.paragraph_level_BiLSTM_label_embedding_MLP_pre import MyModel
from train_valid_test.test_paragraph_level_model_long_short import long_short_get
from train_valid_test.train_valid_paragraph_level_model import train_and_valid_special_deepcopy

import argparse
parser = argparse.ArgumentParser(description='para transfer')
parser.add_argument('--EPOCHs', type=int, default=20)
parser.add_argument('--DROPOUT', type=float, default=0.5)
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--LEARN_RATE', type=float, default=1e-3)
parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-4)
parser.add_argument('--fold_num', type=int, default=0)
parser.add_argument('--cheat', type=str, default='False')
parser.add_argument('--mask_p', type=float, default=0.0)
parser.add_argument('--bilstm_1_grad', type=int, default=1)  # default is True: use grad
parser.add_argument('--if_control_loss', type=int, default=0)  # default is False
parser.add_argument('--if_use_memory', type=int, default=0)  # default is False
args = parser.parse_args()
print(args)

EPOCHs = args.EPOCHs
DROPOUT = args.DROPOUT
BATCH_SIZE = args.BATCH_SIZE
LEARN_RATE = args.LEARN_RATE
WEIGHT_DECAY = args.WEIGHT_DECAY
fold_num = args.fold_num
cheat = args.cheat
mask_p = args.mask_p
bilstm_1_grad = args.bilstm_1_grad
if_control_loss = args.if_control_loss
if_use_memory = args.if_use_memory

if __name__ == '__main__':

    test_f1_list = []
    test_acc_list = []
    valid_best_f1_list = []
    valid_best_acc_list = []

    for t in range(1):
        print("\n\n\n\ntime=", t)

        train_data_list, valid_data_list, test_data_list = get_data(if_do_embedding=True,
                                                                    stanford_path='stanford-corenlp-4.3.1',
                                                                    random_seed=fold_num)
        train_batch_list = get_train_batch_list(train_data_list, BATCH_SIZE, each_data_len=0)

        if int(if_use_memory) == 0:  # use pre_model do ex_pre_label
            pre_model_id = random.randint(0, 10000)
            sentence_level_best_model = run_sentence_level_BiLSTM_ex.main(train_data_list, valid_data_list, test_data_list, pre_model_id, two_C=False)
            train_data_memory = None
        else:
            print('use memory sim, no need pre_bilstm')
            sentence_level_best_model = None
            train_data_memory = get_memory(stanford_path='stanford-corenlp-4.3.1')
            print("already get memory")
            print("len(train_data_memory): ", len(train_data_memory))

        ex_model_extra = None
        if mask_p == 0.1:
            print("will get ex_model_extra")
            pre_model_id = random.randint(0, 10000)
            ex_model_extra = run_sentence_level_BiLSTM_ex.main(train_data_list, valid_data_list, test_data_list, pre_model_id, two_C=True)

        model = MyModel(dropout=DROPOUT,
                        stanford_path='stanford-corenlp-4.3.1',
                        # pre_model_path='models/' + str(pre_model_id) + '_model_sentence_level_BiLSTM_extra.pt',
                        ex_model=sentence_level_best_model,
                        cheat=cheat,
                        mask_p=mask_p,
                        ex_model_grad=bilstm_1_grad,
                        if_control_loss=if_control_loss,
                        if_use_memory=if_use_memory,
                        train_data_memory=train_data_memory,
                        ex_model_extra=ex_model_extra).cuda()
        temp_best_model = MyModel(dropout=DROPOUT,
                                  stanford_path='stanford-corenlp-4.3.1',
                                  # pre_model_path='models/' + str(pre_model_id) + '_model_sentence_level_BiLSTM_extra.pt',
                                  ex_model=sentence_level_best_model,
                                  cheat=cheat,
                                  mask_p=mask_p,
                                  ex_model_grad=bilstm_1_grad,
                                  if_control_loss=if_control_loss,
                                  if_use_memory=if_use_memory,
                                  train_data_memory=train_data_memory,
                                  ex_model_extra=ex_model_extra).cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)
        print('paragraph level model init done')

        best_epoch, best_model, best_macro_Fscore, best_acc = train_and_valid_special_deepcopy(model, temp_best_model, optimizer, train_batch_list,
                                                                                               valid_data_list, EPOCHs, with_raw_text=True)
        # torch.save(best_model, 'output/model_paragraph_level_BiLSTM_label_embedding_MLP_pre_' + str(cheat) + '_' + str(mask_p) + '.pt')
        print("best_epoch: ", best_epoch, best_macro_Fscore, best_acc)

        f1_score, acc = test_model(test_data_list, best_model, with_raw_text=True)
        long_short_get(test_data_list, best_model, with_raw_text=True)

        test_f1_list.append(f1_score)
        test_acc_list.append(acc)
        valid_best_f1_list.append(best_macro_Fscore)
        valid_best_acc_list.append(best_acc)

    # ###############################
    print("test f1:  ", np.mean(np.array(test_f1_list)), test_f1_list)
    print("test ass: ", np.mean(np.array(test_acc_list)), test_acc_list)
    print("valid f1: ", np.mean(np.array(valid_best_f1_list)), valid_best_f1_list)
    print("valid acc:", np.mean(np.array(valid_best_acc_list)), valid_best_acc_list)

    if os.path.exists('output/paragraph_level_BiLSTM/temp/' + str(args) + '/valid_f1') == False:
        os.mkdir(path='output/paragraph_level_BiLSTM/temp/' + str(args) + '/valid_f1')
    if os.path.exists('output/paragraph_level_BiLSTM/temp/' + str(args) + '/valid_acc') == False:
        os.mkdir(path='output/paragraph_level_BiLSTM/temp/' + str(args) + '/valid_acc')
    if os.path.exists('output/paragraph_level_BiLSTM/temp/' + str(args) + '/test_f1') == False:
        os.mkdir(path='output/paragraph_level_BiLSTM/temp/' + str(args) + '/test_f1')
    if os.path.exists('output/paragraph_level_BiLSTM/temp/' + str(args) + '/test_acc') == False:
        os.mkdir(path='output/paragraph_level_BiLSTM/temp/' + str(args) + '/test_acc')

    open('output/paragraph_level_BiLSTM/temp/' + str(args) + '/test_acc/' + str(np.mean(np.array(test_acc_list))) + '.txt', 'w')
    open('output/paragraph_level_BiLSTM/temp/' + str(args) + '/test_f1/' + str(np.mean(np.array(test_f1_list))) + '.txt', 'w')
    open('output/paragraph_level_BiLSTM/temp/' + str(args) + '/valid_acc/' + str(np.mean(np.array(valid_best_acc_list))) + '.txt', 'w')
    open('output/paragraph_level_BiLSTM/temp/' + str(args) + '/valid_f1/' + str(np.mean(np.array(valid_best_f1_list))) + '.txt', 'w')
