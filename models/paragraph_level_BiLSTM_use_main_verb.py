import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial_1, if_use_ex_initial_2, freeze):
        super(MyModel, self).__init__()
        print("paragraph level BiLSTM main verb for label embedding")

        self.main_verb_contribution = torch.load('resource/statistic_dict_plus.pt')

        self.random_seed = random_seed

        self.dropout = nn.Dropout(p=dropout)

        self.label_embedding = nn.Parameter(torch.randn(7, 300), requires_grad=True)

        self.BiLSTM_1 = nn.LSTM(input_dim, 300 // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.BiLSTM_2 = nn.LSTM(300, 300 // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)

        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

        self.weight_average_label_embedding_ffn = nn.Sequential(
            nn.Linear(300, 7),
            nn.LogSoftmax()
        )

        self.merge_mlp = nn.Sequential(
            nn.Linear(643, 500),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(500, 400),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(400, 343),
            nn.ReLU()
        )

    def forward(self, sentences_list, main_verb_list, main_verb_position_list):  # [4*3336, 7*336, 1*336]
        sentence_embeddings_list = torch.zeros(len(sentences_list), 300).cuda()
        for i in range(len(sentences_list)):
            main_verb = main_verb_list[i]
            main_verb_position = main_verb_position_list[i]
            sentence = self.dropout(sentences_list[i])

            word_embeddings_list = self.merge(sentence, main_verb, main_verb_position)
            word_embeddings_list = word_embeddings_list.unsqueeze(0)  # 1 * sentence_len * 343

            word_embeddings_output, _ = self.BiLSTM_1(word_embeddings_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))  # 1 * sentence_len * 300
            sentence_embedding = torch.max(word_embeddings_output[0, :, :], 0)[0]  # size = 300
            sentence_embeddings_list[i, :] = sentence_embedding

        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 300
        sentence_embeddings_list = self.dropout(sentence_embeddings_list)

        sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))
        sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300
        sentence_embeddings_output = self.dropout(sentence_embeddings_output)

        output_2 = self.softmax(self.hidden2tag(sentence_embeddings_output))  # 3 * 7

        pre_labels_list_2 = []
        for i in range(output_2.shape[0]):
            pre_labels_list_2.append(int(torch.argmax(output_2[i])))

        return [pre_labels_list_2, []], [output_2, _], sentence_embeddings_output

    def merge(self, word_embeddings_list, main_verb, main_verb_position):
        if main_verb_position == -1:
            return word_embeddings_list
        elif main_verb == 'None':
            main_verb_contribution = torch.tensor(np.array([0.0, 0.0, 0.0, 1/2, 1/2, 0, 0])).to(torch.float32).unsqueeze(0).cuda()  # 1 * 7
            average_label_embedding = torch.einsum('ij, jk->ik', main_verb_contribution, self.label_embedding).squeeze(0)  # size is 300
            merge_embedding = torch.cat((word_embeddings_list[main_verb_position], average_label_embedding), dim=0)  # size is 643
            merge_embedding = self.merge_mlp(merge_embedding)  # size is 343
            word_embeddings_list[main_verb_position, :] = merge_embedding
            return word_embeddings_list
        elif main_verb not in list(self.main_verb_contribution.keys()):
            return word_embeddings_list
        elif self.main_verb_contribution[main_verb][0][7] < 5:
            return word_embeddings_list
        else:
            main_verb_contribution = torch.tensor(np.array(self.main_verb_contribution[main_verb][1])).to(torch.float32).unsqueeze(0).cuda()  # 1 * 7
            main_verb_contribution[0, 0] = main_verb_contribution[0, 0] / 2
            average_label_embedding = torch.einsum('ij, jk->ik', main_verb_contribution, self.label_embedding).squeeze(0)  # size is 300
            merge_embedding = torch.cat((word_embeddings_list[main_verb_position], average_label_embedding), dim=0)  # size is 643
            merge_embedding = self.merge_mlp(merge_embedding)  # size is 343
            word_embeddings_list[main_verb_position, :] = merge_embedding
            return word_embeddings_list
