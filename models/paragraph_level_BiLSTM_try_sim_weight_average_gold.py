import torch
import torch.nn as nn
from torch.autograd import Variable


print("paragraph level BiLSTM try sim weight average gold")


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.BiLSTM_1 = nn.LSTM(300,
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
        self.log_softmax = nn.LogSoftmax()

        self.softmax = nn.Softmax()  # used as weight average

        self.sim_softmax = nn.Softmax(dim=0)
        self.reset_num = 0
        self.correct_representation_list = torch.tensor([[0.0 for _ in range(300)] for __ in range(7)]).cuda()  # each class a correct representation
        self.correct_num_list = [1] * 7
        self.last_epoch_correct_representation_list = torch.tensor([[0.0 for __ in range(300)] for _ in range(7)]).cuda()  # each class a correct representation
        self.sim_matrix = [[0 for _ in range(7)] for __ in range(7)]

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        sentence_embeddings_list = []
        for sentence in sentences_list:
            word_embeddings_list = sentence.unsqueeze(0).cuda()  # 1 * sentence_len * 336

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

        log_softmax_output = self.log_softmax(output)  # 3 * 7

        softmax_output = self.softmax(output)  # size is 3 * 7

        pre_labels_list = []
        for i in range(log_softmax_output.shape[0]):
            pre_labels_list.append(int(torch.argmax(log_softmax_output[i])))

        loss = 0
        for j in range(len(gold_labels_list)):
            gold_label = gold_labels_list[j]
            pre_label = pre_labels_list[j]
            loss += -log_softmax_output[j][gold_label]

            sentence_embedding_new = sentence_embeddings_output[j, :]  # size is 300
            if self.reset_num > 1:
                # ###################################################### check if label is related with sim
                sim_list = []
                for i in range(7):
                    sim_list.append(torch.cosine_similarity(sentence_embedding_new,
                                                            self.last_epoch_correct_representation_list[i, :],
                                                            dim=0))
                # sim_list = self.sim_softmax(torch.tensor(sim_list))
                for i in range(7):
                    self.sim_matrix[gold_label][i] += sim_list[i]
                # ###########################################################################################################

                if pre_label != gold_label:
                    sim_loss = torch.cosine_similarity(sentence_embedding_new,
                                                       self.last_epoch_correct_representation_list[pre_label, :],
                                                       dim=0)
                    loss += sim_loss

            if pre_label == gold_label:
                self.correct_representation_list[gold_label] = self.correct_representation_list[gold_label] + \
                                                               sentence_embedding_new * softmax_output[j][gold_label]
                self.correct_num_list[gold_label] += softmax_output[j][gold_label]

        return pre_labels_list, loss

    def reset(self):
        if self.reset_num > 1:
            print(torch.tensor(self.sim_matrix).int())

        self.reset_num += 1
        print("self.reset_num: ", self.reset_num)

        for i in range(7):
            self.correct_representation_list[i, :] = self.correct_representation_list[i, :] / self.correct_num_list[i]
            # print(self.correct_representation_list[i, :])

        self.last_epoch_correct_representation_list = self.correct_representation_list

        self.correct_num_list = [1] * 7
        self.correct_representation_list = torch.tensor([[0.0 for _ in range(300)] for _ in range(7)]).cuda()  # each class a gold representation

        self.correct_representation_list = self.correct_representation_list.detach()
        self.last_epoch_correct_representation_list = self.last_epoch_correct_representation_list.detach()

        self.sim_matrix = [[0 for _ in range(7)] for __ in range(7)]

        # if self.correct_representation_list.requires_grad:
        #     print("retain_grad")
        #     self.correct_representation_list = self.correct_representation_list.retain_grad()
        #     self.last_epoch_correct_representation_list = self.last_epoch_correct_representation_list.retain_grad()