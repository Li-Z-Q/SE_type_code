import torch
import torch.nn as nn
from torch.autograd import Variable


print("sentence level BiLSTM extra")


class MyModel(nn.Module):
    def __init__(self, dropout=0, two_C=False):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.BiLSTM = nn.LSTM(300,
                              300 // 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True,
                              dropout=dropout)

        print("two_C: ", two_C)
        if two_C:
            self.hidden2tag = nn.Linear(300, 2)
        else:
            self.hidden2tag = nn.Linear(300, 7)

        self.log_softmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, word_embeddings_list, gold_label):
        word_embeddings_list = word_embeddings_list.unsqueeze(0).cuda()  # 1 * sentence_len * 300

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        BiLSTM_output, _ = self.BiLSTM(word_embeddings_list, init_hidden)  # 1 * sentence_len * 300

        sentence_embedding = torch.max(BiLSTM_output, 1)[0]  # 1 * 300

        hidden_output = self.hidden2tag(sentence_embedding)  # 1 * 7

        output = self.log_softmax(hidden_output)  # 1 * 7
        output = output.squeeze(0)

        softmax_output = self.softmax(hidden_output)
        softmax_output = softmax_output.squeeze(0)  # size is 7, uesed to representantion believeable

        pre_label = int(torch.argmax(output))
        loss = -output[gold_label]

        return pre_label, loss, sentence_embedding, softmax_output

    def load_model(self, path):
        return torch.load(path)
