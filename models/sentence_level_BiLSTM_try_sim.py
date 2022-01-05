import copy
import torch
import torch.nn as nn
from torch.autograd import Variable


print("sentence level BiLSTM try sim")


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.BiLSTM = nn.LSTM(300,
                              300 // 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True,
                              dropout=dropout)
        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

        self.reset_num = 0

        self.correct_representation_list = torch.tensor([[0.0 for _ in range(300)] for _ in range(7)]).cuda()  # each class a correct representation
        self.correct_num_list = [1] * 7
        self.last_epoch_correct_representation_list = torch.tensor([[0.0 for _ in range(300)] for _ in range(7)]).cuda()  # each class a correct representation

    def forward(self, word_embeddings_list, gold_label):

        word_embeddings_list = word_embeddings_list.unsqueeze(0).cuda()  # 1 * sentence_len * 300

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        BiLSTM_output, _ = self.BiLSTM(word_embeddings_list, init_hidden)  # 1 * sentence_len * 300

        sentence_embedding = torch.max(BiLSTM_output, 1)[0]  # 1 * 300

        hidden_output = self.hidden2tag(sentence_embedding)  # 1 * 7

        output = self.softmax(hidden_output)  # 1 * 7
        output = output.squeeze(0)

        pre_label = int(torch.argmax(output))
        loss = -output[gold_label]

        hidden_output = hidden_output.squeeze(0)  # size is 7
        sentence_embedding = sentence_embedding.squeeze(0)  # size is 300

        if self.reset_num > 1:
            if pre_label != gold_label:
                sim_loss = torch.cosine_similarity(sentence_embedding, self.last_epoch_correct_representation_list[gold_label, :], dim=0)
                loss += sim_loss

        if pre_label == gold_label:
            self.correct_representation_list[gold_label] = self.correct_representation_list[gold_label] + sentence_embedding
            self.correct_num_list[gold_label] += 1

        return pre_label, loss

    def reset(self):

        self.reset_num += 1
        print("self.reset_num: ", self.reset_num)

        for i in range(7):
            self.correct_representation_list[i, :] = self.correct_representation_list[i, :] / self.correct_num_list[i]

        self.last_epoch_correct_representation_list = self.correct_representation_list

        self.correct_num_list = [1] * 7
        self.correct_representation_list = torch.tensor([[0.0 for _ in range(300)] for _ in range(7)]).cuda()  # each class a gold representation

        self.correct_representation_list = self.correct_representation_list.detach()
        self.last_epoch_correct_representation_list = self.last_epoch_correct_representation_list.detach()

#

