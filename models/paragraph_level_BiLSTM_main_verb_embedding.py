import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial_1, if_use_ex_initial_2, freeze):
        super(MyModel, self).__init__()
        print("paragraph level BiLSTM main verb for label embedding")

        self.main_verb_contribution = torch.load('resource/statistic_dict_plus_test.pt')

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
            nn.Linear(600, 500),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(500, 400),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(400, 300),
            nn.ReLU()
        )

    def forward(self, sentences_list, main_verb_list):  # [4*3336, 7*336, 1*336]
        sentence_embeddings_list = torch.zeros(len(sentences_list), 300).cuda()
        weight_average_label_embedding_list = []
        for i in range(len(sentences_list)):
            sentence = self.dropout(sentences_list[i])
            word_embeddings_list = sentence.unsqueeze(0)  # 1 * sentence_len * 343
            word_embeddings_output, _ = self.BiLSTM_1(word_embeddings_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))  # 1 * sentence_len * 300
            sentence_embedding = torch.max(word_embeddings_output[0, :, :], 0)[0]  # size = 300

            merge_embedding, weight_average_label_embedding = self.merge(sentence_embedding, main_verb_list[i])
            weight_average_label_embedding = self.weight_average_label_embedding_ffn(weight_average_label_embedding)  # get 7
            weight_average_label_embedding_list.append(weight_average_label_embedding)
            sentence_embeddings_list[i, :] = merge_embedding

        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 300
        sentence_embeddings_list = self.dropout(sentence_embeddings_list)

        sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))
        sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300
        sentence_embeddings_output = self.dropout(sentence_embeddings_output)

        output_2 = self.softmax(self.hidden2tag(sentence_embeddings_output))  # 3 * 7

        pre_labels_list_2 = []
        for i in range(output_2.shape[0]):
            pre_labels_list_2.append(int(torch.argmax(output_2[i])))

        return [pre_labels_list_2, []], [output_2, weight_average_label_embedding_list], sentence_embeddings_output

    def merge(self, sentence_embedding, main_verb):
        # print(main_verb)
        # print(self.main_verb_contribution[main_verb])
        if main_verb not in list(self.main_verb_contribution.keys()) or main_verb == 'None':
            weight_average_label_embedding = torch.mean(self.label_embedding, dim=0)
            return sentence_embedding, weight_average_label_embedding
        elif self.main_verb_contribution[main_verb][0][7] < 5:
            weight_average_label_embedding = torch.mean(self.label_embedding, dim=0)
            return sentence_embedding, weight_average_label_embedding
        else:
            main_verb_contribution = torch.tensor(np.array(self.main_verb_contribution[main_verb][1])).to(torch.float32).cuda()
            main_verb_contribution = main_verb_contribution.unsqueeze(0)  # 1 * 7
            weight_average_label_embedding = torch.einsum('ij, jk->ik', main_verb_contribution, self.label_embedding)
            weight_average_label_embedding = weight_average_label_embedding.squeeze(0)  # size is 300
            merge_embedding = torch.cat((sentence_embedding, weight_average_label_embedding), dim=0)
            merge_embedding = self.dropout(merge_embedding)
            merge_embedding = self.merge_mlp(merge_embedding)  # from 600 -> 300
            merge_embedding = self.dropout(merge_embedding)
            return merge_embedding, weight_average_label_embedding
