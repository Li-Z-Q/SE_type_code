import os
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/model')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)

import warnings
warnings.filterwarnings('ignore')

import copy
import torch
from torch import optim
from model.paragraph_level_BERT import MyModel
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

        # print("before train: ", torch.cuda.memory_allocated(0))
        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                sentences_list = []
                gold_labels_list = []
                for sentence, label in zip(train_data[0], train_data[1]):
                    if label != 7:
                        gold_labels_list.append(label)
                        sentences_list.append(sentence)

                _, loss = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7

                batch_loss += loss

            if batch_loss != 0:
                batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                sentences_list = []
                gold_labels_list = []
                for sentence, label in zip(valid_data[0], valid_data[1]):
                    if label != 7:
                        gold_labels_list.append(label)
                        sentences_list.append(sentence)

                pre_labels_list, _ = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7

                for i in range(len(gold_labels_list)):
                    useful_target_Y_list.append(gold_labels_list[i])
                    useful_predict_Y_list.append(pre_labels_list[i])

        # ################################### print and save model ##############################
        tmp_macro_Fscore = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_Fscore > best_macro_Fscore:
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_macro_Fscore = tmp_macro_Fscore

    return best_epoch, best_model, best_macro_Fscore


EPOCHs = 20
DROPOUT = 0.5
BATCH_SIZE = 4
LEARN_RATE = 1e-5
WEIGHT_DECAY = 1e-4
train_data_list, valid_data_list = get_data(if_do_embedding=False, stanford_path='stanford-corenlp-4.3.1')

if __name__ == '__main__':

    model = MyModel(dropout=DROPOUT).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

    best_epoch, best_model, best_macro_Fscore = train_and_valid()
    torch.save(best_model, 'output/model_paragraph_level_BERT.pt')
    print("best_epoch: ", best_epoch, best_macro_Fscore)
