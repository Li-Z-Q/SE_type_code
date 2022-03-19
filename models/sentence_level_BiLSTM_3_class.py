import torch
import torch.nn as nn
from torch.autograd import Variable


print("sentence level BiLSTM")


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial):
        super(MyModel, self).__init__()

        self.random_seed = random_seed
        self.if_use_ex_initial = if_use_ex_initial

        self.dropout = nn.Dropout(p=dropout)

        if self.if_use_ex_initial:
            self.BiLSTM = self.load()
        else:
            self.BiLSTM = nn.LSTM(input_dim,
                                  300 // 2,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True,
                                  dropout=dropout)
        self.hidden2tag = nn.Linear(300, 3)
        self.softmax = nn.LogSoftmax()

    def forward(self, word_embeddings_list):
        word_embeddings_list = word_embeddings_list.unsqueeze(0)  # 1 * sentence_len * 348
        # word_embeddings_list = self.dropout(word_embeddings_list)

        if not self.if_use_ex_initial:
            BiLSTM_output, _ = self.BiLSTM(word_embeddings_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))  # 1 * sentence_len * 300
            sentence_embedding = torch.max(BiLSTM_output, 1)[0]  # 1 * 300
        else:
            _, _, sentence_embedding = self.BiLSTM(word_embeddings_list)
            sentence_embedding = sentence_embedding.unsqueeze(0)  # 1 * 300

        # sentence_embedding = self.dropout(sentence_embedding)
        output = self.softmax(self.hidden2tag(sentence_embedding))  # 1 * 7
        output = output.squeeze(0)  # size is 7

        pre_label = int(torch.argmax(output))

        return pre_label, output, sentence_embedding.squeeze(0)

    def save(self):
        torch.save(self, 'models/model_sentence_level_BiLSTM_3_class_' + str(self.random_seed) + '.pt')

    def load(self):
        return torch.load('models/model_sentence_level_BiLSTM_3_class' + str(self.random_seed) + '.pt')
