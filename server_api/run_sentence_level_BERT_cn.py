import os
import random
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/model')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/pre_train_cn')
print(sys.path)

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import optim
from transformers import BertTokenizer
from model.sentence_level_BERT import MyModel
from tools.devide_train_batch import get_train_batch_list
from train_valid_test.train_valid_sentence_level_model import train_and_valid
from train_valid_test.test_sentence_level_model import test_model


pre_train_path = 'pre_train_cn'


def get_data():
    print('will deal data')
    tokenizer = BertTokenizer.from_pretrained(pre_train_path)
    data_list = []
    raw_file = open('data/weibo_events_annotate.txt', 'r', encoding='utf-8')
    for line in raw_file:
        raw_data = line.strip('\n').split(', ')
        # print(raw_data[0])
        data_list.append([raw_data[0], int(raw_data[1]), tokenizer(raw_data[0], return_tensors="pt").input_ids.cuda()])
    random.shuffle(data_list)

    print("len(data_list): ", len(data_list))

    return data_list[:int(0.8 * len(data_list))], data_list[int(0.8 * len(data_list)):int(0.9 * len(data_list))], data_list[int(0.9 * len(data_list)):]


if __name__ == '__main__':
    train_data_list, valid_data_list, test_data_list = get_data()
    train_batch_list = get_train_batch_list(train_data_list, BATCH_SIZE=8, each_data_len=1)

    model = MyModel(dropout=0.5, num_labels=2, pre_train_path=pre_train_path).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    best_epoch, best_model, best_macro_Fscore, best_acc = \
        train_and_valid(model, optimizer, train_batch_list, valid_data_list, total_epoch=10)
    best_model.save('./BERT_1_cn.pt')
    print("\n\nbest_epoch: ", best_epoch, best_macro_Fscore, best_acc)

    f1_score, acc = test_model(test_data_list, best_model)