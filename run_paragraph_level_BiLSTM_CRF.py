import os
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/model')
sys.path.append(os.getcwd() + '/tools')
print(sys.path)

import warnings
warnings.filterwarnings('ignore')

import copy
import torch
from torch import optim
from model.paragraph_level_BiLSTM_CRF import MyModel
from tools.get_paragraph_level_data import get_data
from tools.devide_train_batch import get_train_batch_list
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid():
    train_batch_list = get_train_batch_list(train_data_list, BATCH_SIZE, each_data_len=0)

    best_model = None
    best_epoch = None
    best_macro_Fscore = -1
    for epoch in range(EPOCHs):
        print('\n\nepoch ' + str(epoch) + '/' + str(EPOCHs))

        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                sentence_embeddings_list = []
                labels_list = []
                for sentence_embedding, label in zip(train_data[2], train_data[1]):
                    if label != 7:  # before using CRF, so must delete label==7
                        labels_list.append(label)
                        sentence_embeddings_list.append(sentence_embedding.cuda())

                _, loss = model.forward(sentence_embeddings_list, torch.tensor(labels_list).cuda())  # sentence_num * 7

                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                sentence_embeddings_list = valid_data[2]
                labels_list = valid_data[1]

                output, _ = model.forward(sentence_embeddings_list, labels_list=None)  # 1 * sentence_num

                for i in range(len(labels_list)):
                    label = labels_list[i]
                    if label != 7:
                        useful_target_Y_list.append(label)
                        useful_predict_Y_list.append(output[i])

        # ################################### print and save model ##############################
        tmp_macro_Fscore = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_Fscore > best_macro_Fscore:
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_macro_Fscore = tmp_macro_Fscore

    return best_epoch, best_model, best_macro_Fscore


EPOCHs = 50
DROPOUT = 0.5
BATCH_SIZE = 128
LEARN_RATE = 1e-3
WEIGHT_DECAY = 0

train_data_list, valid_data_list = get_data(if_do_embedding=True)

if __name__ == '__main__':

    model = MyModel(dropout=DROPOUT).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

    best_epoch, best_model, best_macro_Fscore = train_and_valid()
    print("best_epoch: ", best_epoch)
