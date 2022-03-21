import torch
import torch.nn as nn
from torch.autograd import Variable


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial):
        super(MyModel, self).__init__()
        print("sentence level Linear 3 class")

        self.random_seed = random_seed

        self.dropout = nn.Dropout(p=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=400),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(400, 360),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(360, 330),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(330, 300),
            nn.ReLU()
        )

        self.hidden2tag = nn.Linear(300, 3)
        self.softmax = nn.LogSoftmax()

    def forward(self, word_embeddings_list):  # sentence_len * 348
        word_embeddings_list = self.dropout(word_embeddings_list)
        ffn_output = self.ffn(word_embeddings_list)  # sentence_len * 300
        sentence_embedding = torch.mean(ffn_output, dim=0).unsqueeze(0)  # 1 * 300

        sentence_embedding = self.dropout(sentence_embedding)
        output = self.softmax(self.hidden2tag(sentence_embedding))  # 1 * 3
        output = output.squeeze(0)  # size is 3

        pre_label = int(torch.argmax(output))

        return pre_label, output, sentence_embedding.squeeze(0)

    def save(self):
        torch.save(self, 'models/model_sentence_level_Linear_3_class_' + str(self.random_seed) + '.pt')

    def load(self):
        return torch.load('models/model_sentence_level_Linear_3_class_' + str(self.random_seed) + '.pt')
