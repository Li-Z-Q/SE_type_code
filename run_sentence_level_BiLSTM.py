import os
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/pre_train')
print(sys.path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import optim
from models.sentence_level_BiLSTM import MyModel
from tools.get_sentence_level_data import get_data
from tools.devide_train_batch import get_train_batch_list
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid():
    train_batch_list = get_train_batch_list(train_data_list, BATCH_SIZE, each_data_len=1)

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
                gold_label = train_data[1]
                word_embeddings_list = train_data[2]  # sentence_len * 343

                pre_label, loss = model.forward(word_embeddings_list.cuda(), gold_label)  # 1 * 7

                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                gold_label = valid_data[1]
                word_embeddings_list = valid_data[2]  # sentence_len * 300

                pre_label, loss = model.forward(word_embeddings_list.cuda(), gold_label)  # 1 * 7

                useful_target_Y_list.append(gold_label)
                useful_predict_Y_list.append(pre_label)

        # ################################### print and save models ##############################
        tmp_macro_Fscore = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_Fscore > best_macro_Fscore:
            best_epoch = epoch
            best_model = model
            best_macro_Fscore = tmp_macro_Fscore

    return best_epoch, best_model, best_macro_Fscore


EPOCHs = 40
DROPOUT = 0.5
BATCH_SIZE = 128
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-4
train_data_list, valid_data_list, test_data_list = get_data(if_do_embedding=True, stanford_path='stanford-corenlp-4.3.1')

model = MyModel(dropout=DROPOUT).cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)


if __name__ == '__main__':

    best_epoch, best_model, best_macro_Fscore = train_and_valid()
    torch.save(best_model, 'output/model_sentence_level_BiLSTM.pt')
    print("best_epoch: ", best_epoch, best_macro_Fscore)

