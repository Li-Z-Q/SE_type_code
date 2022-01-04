import torch
import torch.nn as nn
from torchcrf import CRF
from torch.autograd import Variable


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.BiLSTM_1 = nn.LSTM(343,
                                300 // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)
        self.BiLSTM_2 = nn.LSTM(300,
                                300 // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)

        self.hidden2tag = nn.Linear(300, 7)
        # self.softmax = nn.Softmax()
        self.crf = CRF(num_tags=7, batch_first=True)

    def forward(self, sentences_list, labels_list):

        sentence_embeddings_list = []
        for sentence in sentences_list:
            sentence = self.dropout(sentence)

            word_embeddings_list = sentence.unsqueeze(0)  # 1 * sentence_len * 336

            init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
            word_embeddings_output, _ = self.BiLSTM_1(word_embeddings_list, init_hidden)  # 1 * sentence_len * 300

            sentence_embedding = torch.max(word_embeddings_output[0, :, :], 0)[0]  # size = 300
            sentence_embeddings_list.append(sentence_embedding)
        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 300
        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 300

        sentence_embeddings_list = self.dropout(sentence_embeddings_list)

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, init_hidden)  # 1 * sentence_num * 300
        sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300

        pro_matrix = self.hidden2tag(sentence_embeddings_output)  # sentence_num * 7
        # output = self.softmax(output)

        pro_matrix = pro_matrix.unsqueeze(0)  # 1 * sentence_num * 7

        if labels_list != None:
            labels_list = labels_list.unsqueeze(0)  # 1 * sentence_num
            loss = -self.crf(pro_matrix, labels_list)
        else:
            loss = 0

        output = self.crf.decode(pro_matrix)  # 1 * sentence_num

        output = output[0]  # is a list, len = sentence_num

        # print(output)
        # print(labels_list)

        return output, loss