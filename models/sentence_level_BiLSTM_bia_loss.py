import torch
import torch.nn as nn
from torch.autograd import Variable


print("sentence level BiLSTM bia loss")


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.BiLSTM = nn.LSTM(343,
                              300 // 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True,
                              dropout=dropout)
        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

    def forward(self, word_embeddings_list, gold_label):
        word_embeddings_list = word_embeddings_list.unsqueeze(0).cuda()  # 1 * sentence_len * 300

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        BiLSTM_output, _ = self.BiLSTM(word_embeddings_list, init_hidden)  # 1 * sentence_len * 300

        sentence_embedding = torch.max(BiLSTM_output, 1)[0]  # 1 * 300

        output = self.hidden2tag(sentence_embedding)  # 1 * 7

        output = self.softmax(output)  # 1 * 7
        output = output.squeeze(0)

        pre_label = int(torch.argmax(output))
        loss = -output[gold_label]

        if gold_label == 0:
            loss += output[1]
            loss += output[3]
        if gold_label == 1:
            loss += output[0]
        if gold_label == 2:
            loss += output[1]
        if gold_label == 3:
            loss += output[0]
            loss += output[1]
        if gold_label == 4:
            loss += output[0]
            loss += output[1]
            loss += output[3]
        if gold_label == 5:
            loss += output[0]

        return pre_label, loss
