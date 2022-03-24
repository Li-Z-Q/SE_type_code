import torch
import torch.nn as nn
from torch.autograd import Variable
from tools.print_evaluation_result import print_evaluation_result
from models.sentence_level_BiLSTM import MyModel as SentenceLevelModelBase


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial_1, if_use_ex_initial_2, freeze):
        super(MyModel, self).__init__()
        print("paragraph level BiLSTM label embedding")

        self.random_seed = random_seed
        self.if_use_ex_initial_1 = if_use_ex_initial_1
        self.if_use_ex_initial_2 = if_use_ex_initial_2

        self.dropout = nn.Dropout(p=dropout)

        self.label_embedding = nn.Parameter(torch.randn(7, 300), requires_grad=True)

        if self.if_use_ex_initial_2:
            self.BiLSTM_1_2 = self.load()
        else:
            self.BiLSTM_1 = self.load()
            if freeze:
                for param in self.BiLSTM_1.named_parameters():
                    param[1].requires_grad = False

            self.BiLSTM_2 = nn.LSTM(300, 300 // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)

        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

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

    def forward(self, sentences_list):  # [4*3336, 7*336, 1*336]
        output_1_list = []
        ex_pre_label_list = []
        if self.if_use_ex_initial_2:
            pre_labels_list, output, sentence_embeddings_output = self.BiLSTM_1_2(sentences_list)
            return pre_labels_list, output, sentence_embeddings_output
        else:  # self.if_use_ex_initial_1
            sentence_embeddings_list = []

            for sentence_ in sentences_list:
                sentence = self.dropout(sentence_)
                word_embeddings_list = sentence  # sentence_len * 336
                ex_pre_label, _, _ = self.BiLSTM_1(word_embeddings_list)
                ex_pre_label_list.append(ex_pre_label)
            for i in range(len(sentences_list)):
                sentence = self.dropout(sentences_list[i])
                word_embeddings_list = sentence  # sentence_len * 336
                ex_pre_label, output_1, sentence_embedding = self.BiLSTM_1(word_embeddings_list)

                merge_embedding = self.merge(sentence_embedding, self.label_embedding[ex_pre_label, :])
                # merge_embedding = self.merge_contextual(sentence_embedding, ex_pre_label_list, i)

                sentence_embeddings_list.append(merge_embedding)  # merge_embedding size is 300
                output_1_list.append(output_1)

            sentence_embeddings_list = torch.stack(sentence_embeddings_list).unsqueeze(0)  # 1 * sentence_num * 300
            sentence_embeddings_list = self.dropout(sentence_embeddings_list)

            sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))
            sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300
            sentence_embeddings_output = self.dropout(sentence_embeddings_output)

            output_2 = self.softmax(self.hidden2tag(sentence_embeddings_output))  # 3 * 7

            pre_labels_list_2 = []
            for i in range(output_2.shape[0]):
                pre_labels_list_2.append(int(torch.argmax(output_2[i])))

            return [pre_labels_list_2, ex_pre_label_list], [output_2, output_1_list], sentence_embeddings_output

    def load(self):
        if self.if_use_ex_initial_1:
            sentence_base_model = SentenceLevelModelBase(input_dim=343, dropout=0.5, random_seed=self.random_seed, if_use_ex_initial=0)
            return sentence_base_model.load()
        elif self.if_use_ex_initial_2:
            return torch.load('models/model_paragraph_level_BiLSTM_base_' + str(self.random_seed) + '.pt')
        else:
            return 0

    def merge(self, sentence_embedding, label_embedding):
        merge_embedding = torch.cat((sentence_embedding, label_embedding), dim=0)
        merge_embedding = self.merge_mlp(merge_embedding)  # from 600 -> 300
        return merge_embedding

    def merge_contextual(self, sentence_embedding, ex_pre_labels_list, position):
        label_embedding_window = torch.randn(5, 300).cuda()
        if position - 2 >= 0:
            label_embedding_window[0, :] = self.label_embedding[ex_pre_labels_list[position-1], :]
        if position - 1 >= 0:
            label_embedding_window[1, :] = self.label_embedding[ex_pre_labels_list[position-1], :]
        if True:
            label_embedding_window[2, :] = self.label_embedding[ex_pre_labels_list[position], :]
        if position + 1 < len(ex_pre_labels_list):
            label_embedding_window[3, :] = self.label_embedding[ex_pre_labels_list[position+1], :]
        if position + 2 < len(ex_pre_labels_list):
            label_embedding_window[4, :] = self.label_embedding[ex_pre_labels_list[position + 1], :]

        label_embedding = torch.mean(label_embedding_window, dim=0)  # get size 300

        merge_embedding = torch.cat((sentence_embedding, label_embedding), dim=0)
        merge_embedding = self.merge_mlp(merge_embedding)  # from 600 -> 300
        return merge_embedding
