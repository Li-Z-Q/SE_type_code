import torch
import torch.nn as nn
from torch.autograd import Variable
from models.sentence_level_BiLSTM import MyModel as SentenceLevelModelBase


print("paragraph level BiLSTM")


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial_1, if_use_ex_initial_2, ex_loss, freeze):
        super(MyModel, self).__init__()

        print('MyModel')

        self.ex_loss = ex_loss
        self.random_seed = random_seed
        self.if_use_ex_initial_1 = if_use_ex_initial_1
        self.if_use_ex_initial_2 = if_use_ex_initial_2

        self.dropout = nn.Dropout(p=dropout)

        if self.if_use_ex_initial_2:
            self.BiLSTM_1_2 = self.load()
        else:
            if self.if_use_ex_initial_1:
                self.BiLSTM_1 = self.load()
                if freeze:
                    for param in self.BiLSTM_1.named_parameters():
                        param[1].requires_grad = False
            else:
                self.BiLSTM_1 = nn.LSTM(input_dim,
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
        self.softmax = nn.LogSoftmax()

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        if self.if_use_ex_initial_2:
            pre_labels_list, loss, sentence_embeddings_output = self.BiLSTM_1_2(sentences_list, gold_labels_list)
        else:
            sentence_embeddings_list = []
            for gold_label, sentence in zip(gold_labels_list, sentences_list):
                word_embeddings_list = sentence.unsqueeze(0)  # 1 * sentence_len * 336

                if self.if_use_ex_initial_1:
                    _, loss_1, sentence_embedding = self.BiLSTM_1(word_embeddings_list, gold_label)
                    sentence_embedding = sentence_embedding.squeeze(0)
                else:
                    init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
                    word_embeddings_output, _ = self.BiLSTM_1(word_embeddings_list, init_hidden)  # 1 * sentence_len * 300
                    sentence_embedding = torch.max(word_embeddings_output[0, :, :], 0)[0]  # size = 300

                sentence_embeddings_list.append(sentence_embedding)

            sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 300
            sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 300

            init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
            sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, init_hidden)  # 1 * sentence_num * 300
            sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300

            output = self.hidden2tag(sentence_embeddings_output)  # 3 * 7
            output = self.softmax(output)  # 3 * 7

            loss_2 = 0
            pre_labels_list = []
            for i in range(output.shape[0]):
                gold_label = gold_labels_list[i]
                if gold_label != 7:
                    loss_2 += -output[i][gold_label]
                    pre_labels_list.append(int(torch.argmax(output[i])))

            loss = loss_2
            if self.if_use_ex_initial_1 and self.ex_loss:
                loss += loss_1

        return pre_labels_list, loss, sentence_embeddings_output

    def save(self):
        torch.save(self, 'models/model_paragraph_level_BiLSTM_base_' + str(self.random_seed) + '.pt')

    def load(self):
        if self.if_use_ex_initial_1:
            sentence_base_model = SentenceLevelModelBase(input_dim=343, dropout=0.5, random_seed=self.random_seed, if_use_ex_initial=0)
            return sentence_base_model.load()
        elif self.if_use_ex_initial_2:
            return torch.load('models/model_paragraph_level_BiLSTM_base_' + str(self.random_seed) + '.pt')
        else:
            return 0


class AuthorModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial_1, if_use_ex_initial_2):
        super(AuthorModel, self).__init__()

        print('AuthorModel')

        self.random_seed = random_seed
        self.if_use_ex_initial_1 = if_use_ex_initial_1
        self.if_use_ex_initial_2 = if_use_ex_initial_2

        self.dropout = nn.Dropout(p=dropout)

        self.BiLSTM_1 = nn.LSTM(input_dim,
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
        self.softmax = nn.LogSoftmax()

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        len_list = [sentence.shape[0] for sentence in sentences_list]

        sentences_list = torch.cat([sentence for sentence in sentences_list], dim=0)  # (4+7+1) * 348
        sentences_list = sentences_list.unsqueeze(0)
        sentences_list = self.dropout(sentences_list)

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        all_word_embeddings_output, _ = self.BiLSTM_1(sentences_list, init_hidden)  # 1 * sentence_len * 300

        start_position = 0
        sentence_embeddings_list = []
        for i in range(len(len_list)):
            sentence_embedding = all_word_embeddings_output[0, start_position:start_position+len_list[i], :]  # sentence_len * 300
            sentence_embedding = torch.max(sentence_embedding, dim=0)[0]  # size is 300
            sentence_embeddings_list.append(sentence_embedding)
            start_position = start_position+len_list[i]

        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 300
        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 300
        sentence_embeddings_list = self.dropout(sentence_embeddings_list)

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, init_hidden)  # 1 * sentence_num * 300
        sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300
        sentence_embeddings_output = self.dropout(sentence_embeddings_output)

        output = self.hidden2tag(sentence_embeddings_output)  # sentence_num * 7
        output = self.softmax(output)  # sentence_num * 7

        loss = 0
        pre_labels_list = []

        for i in range(output.shape[0]):
            gold_label = gold_labels_list[i]
            if gold_label != 7:
                loss += -output[i][gold_label]
                pre_labels_list.append(int(torch.argmax(output[i])))

        return pre_labels_list, loss, sentence_embeddings_output
