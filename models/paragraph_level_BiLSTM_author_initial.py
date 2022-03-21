import torch
import torch.nn as nn
from torch.autograd import Variable
from models.sentence_level_BiLSTM_author import MyModel as SentenceLevelModelBase


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial_1, if_use_ex_initial_2, freeze):
        super(MyModel, self).__init__()
        print("paragraph level BiLSTM author initial")

        self.random_seed = random_seed
        self.if_use_ex_initial_1 = if_use_ex_initial_1
        self.if_use_ex_initial_2 = if_use_ex_initial_2

        self.dropout = nn.Dropout(p=dropout)

        self.BiLSTM_1 = self.load()

        self.BiLSTM_2 = nn.LSTM(300, 300 // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)

        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

    def forward(self, sentences_list):  # [4*3336, 7*336, 1*336]

        [ex_pre_label_list, []], [output_1, []], sentence_embeddings_list = self.BiLSTM_1(sentences_list)  # sentence_embeddings_list size is sentence_len * 300

        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 300
        sentence_embeddings_list = self.dropout(sentence_embeddings_list)

        sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))
        sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300
        sentence_embeddings_output = self.dropout(sentence_embeddings_output)

        output_2 = self.softmax(self.hidden2tag(sentence_embeddings_output))  # 3 * 7

        pre_labels_list_2 = []
        for i in range(output_2.shape[0]):
            pre_labels_list_2.append(int(torch.argmax(output_2[i])))

        return [pre_labels_list_2, ex_pre_label_list], [output_2, []], sentence_embeddings_output

    def load(self):
        sentence_base_model = SentenceLevelModelBase(input_dim=343, dropout=0.5, random_seed=self.random_seed, if_use_ex_initial=0)
        return sentence_base_model.load()