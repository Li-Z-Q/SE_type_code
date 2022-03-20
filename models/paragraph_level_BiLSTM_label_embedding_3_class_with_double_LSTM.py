import torch
import torch.nn as nn
from torch.autograd import Variable
from models.sentence_level_BiLSTM_3_class import MyModel as SentenceLevelModelBase


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial_1, if_use_ex_initial_2, freeze):
        super(MyModel, self).__init__()
        print("paragraph level BiLSTM label embedding 3 class")

        self.random_seed = random_seed
        self.if_use_ex_initial_1 = if_use_ex_initial_1
        self.if_use_ex_initial_2 = if_use_ex_initial_2

        self.dropout = nn.Dropout(p=dropout)

        self.label_embedding = nn.Parameter(torch.randn(7, 300), requires_grad=True)

        self.BiLSTM_1 = self.load_3_class_model()
        if freeze:
            for param in self.BiLSTM_1.named_parameters():
                param[1].requires_grad = False

        self.BiLSTM_1_5 = nn.LSTM(300, 300 // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.BiLSTM_2 = nn.LSTM(300, 300 // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)

        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

        self.merge_mlp = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(643, 300),
            nn.ReLU()
        )

    def forward(self, sentences_list):  # [4*3336, 7*336, 1*336]
        output_1_list = []
        ex_pre_label_list = []

        sentence_embeddings_list = []
        for sentence in sentences_list:
            sentence = self.dropout(sentence)
            word_embeddings_list = sentence  # sentence_len * 336
            ex_pre_label, output_1, sentence_embedding_1 = self.BiLSTM_1(word_embeddings_list)  # sentence_embedding_1 size is 300

            word_embeddings_list_merge = self.merge(sentence_embedding_1, word_embeddings_list).unsqueeze(0)  # 1 * sentence_len * 300
            word_embeddings_list_output, _ = self.BiLSTM_1_5(word_embeddings_list_merge, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))  # 1 * sentence_len * 300
            sentence_embedding = torch.max(word_embeddings_list_output, dim=1)[0].squeeze(0)  # size is 300

            sentence_embeddings_list.append(sentence_embedding)  # sentence_embedding size is 300
            output_1_list.append(output_1)
            ex_pre_label_list.append(ex_pre_label)

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

    def load_3_class_model(self):
        sentence_base_model = SentenceLevelModelBase(input_dim=343, dropout=0.5, random_seed=self.random_seed, if_use_ex_initial=0)
        return sentence_base_model.load()

    def merge(self, sentence_embedding_1, word_embeddings_list):
        sentence_embedding_1 = torch.stack([sentence_embedding_1 for _ in range(word_embeddings_list.shape[0])])  # sentence_len * 300
        merge_embedding = torch.cat((sentence_embedding_1, word_embeddings_list), dim=1)  # sentence_len * 643
        merge_embedding = self.merge_mlp(merge_embedding)  # sentence_len * 300
        return merge_embedding
