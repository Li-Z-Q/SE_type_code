import random

import torch
import gensim
import numpy as np
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from stanfordcorenlp import StanfordCoreNLP
from models.sentence_level_BiLSTM_ex import MyModel as MyModel_ex
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list

print("paragraph level BiLSTM label embedding MLP pre")


def do_label_embedding(stanford_nlp):
    word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('resource/GoogleNews-vectors-negative300.bin', binary=True)
    seType_dict = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']

    label_embedding_list = []
    for label in seType_dict:
        word_embeddings_list = from_sentence_2_word_embeddings_list(label, stanford_nlp, word2vec_vocab).cuda()
        label_embedding = torch.max(word_embeddings_list, dim=0)[0]  # will get size is 300
        label_embedding_list.append(label_embedding)
    label_embedding_list = torch.stack(label_embedding_list)  # 8 * 300

    return label_embedding_list


def do_label_embedding_random():
    label_embedding_list = []
    for i in range(7):
        label_embedding = torch.randn(300)
        label_embedding_list.append(label_embedding)
    label_embedding_list = torch.stack(label_embedding_list)  # 8 * 300
    return label_embedding_list


class MyModel(nn.Module):
    def __init__(self, dropout, stanford_path, ex_model, cheat, mask_p, ex_model_grad, if_control_loss, if_use_memory, train_data_memory, ex_model_extra):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.if_use_memory = bool(if_use_memory)
        self.if_control_loss = bool(if_control_loss)
        print("self.if_use_memory: ", self.if_use_memory)
        print("\nself.if_control_loss: ", self.if_control_loss)

        self.cheat = cheat
        self.mask_p = mask_p
        print("self.cheat: ", self.cheat)
        print("self.mask_p: ", self.mask_p)

        self.train_data_memory = train_data_memory

        self.stanford_nlp = StanfordCoreNLP(stanford_path)

        self.BiLSTM_ex = ex_model
        print("bool(ex_model_grad): ", bool(ex_model_grad))
        for p in self.BiLSTM_ex.parameters():
            p.requires_grad = bool(ex_model_grad)

        self.BiLSTM_extra = ex_model_extra
        for p in self.BiLSTM_extra.parameters():
            p.requires_grad = False
        
        self.BiLSTM_1_independent = nn.LSTM(300,
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

        self.label_embedding_list = nn.Parameter(do_label_embedding(self.stanford_nlp), requires_grad=True)

        self.mlp = nn.Linear(in_features=600, out_features=300)
        self.relu = nn.ReLU()

        self.linear_mask_0_dot_1 = nn.Sequential(
            nn.Linear(300, 350),
            nn.ReLU(),
            nn.Linear(350, 500),
            nn.ReLU(),
            nn.Linear(500, 350),
            nn.ReLU(),
            nn.Linear(350, 300)
        )

        self.hidden2tag = nn.Linear(300, 7)
        self.log_softmax = nn.LogSoftmax()

        self.reset_num = 0

        self.e_pre_labels_list = []
        self.e_gold_labels_list = []

        self.used_e_pre_label_list = []
        self.used_e_gold_labels_list = []

        self.valid_flag = False

    def forward(self, sentences_list_with_without_raw, gold_labels_list):  # [4*3336, 7*336, 1*336]

        sentences_list = sentences_list_with_without_raw[0]
        raw_sentences_list = sentences_list_with_without_raw[1]

        loss = 0
        ex_total_loss = 0
        extra_pre_labels_list = []
        sentence_embeddings_list = []
        for i in range(len(sentences_list)):

            # get sentence embedding
            sentence = sentences_list[i]
            word_embeddings_list = sentence.cuda()  # sentence_len * 300

            ex_gold_label = gold_labels_list[i]
            extra_gold_label = int(ex_gold_label != 0)  # only 2 class

            if self.cheat == "False":
                if self.mask_p == 0.0:  # without extra, only choose according to ex_pre_label
                    ex_pre_label, ex_loss, sentence_embedding, softmax_output = self.BiLSTM_ex(word_embeddings_list, ex_gold_label)  # softmax_output size is 7
                    sentence_embedding = sentence_embedding.squeeze(0)  # size is 300
                    if self.if_control_loss:
                        ex_total_loss += ex_loss

                    # selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    selected_label_embedding = torch.einsum("ij, i->ij", self.label_embedding_list, softmax_output)  # 7 * 300
                    selected_label_embedding = torch.mean(selected_label_embedding, dim=0)  # 300
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, extra_gold_label)
                    sentence_embeddings_list.append(joint_sentence_embedding)

                    if self.valid_flag:
                        self.e_pre_labels_list.append(ex_pre_label)
                        self.e_gold_labels_list.append(ex_gold_label)

                if self.mask_p == 0.1:  # use extra_pre_label == 0:, only have two class
                    _, ex_loss, sentence_embedding, _ = self.BiLSTM_ex(word_embeddings_list, ex_gold_label)  # sentence_embedding is 1 * 300
                    sentence_embedding = sentence_embedding.squeeze(0)  # size is 300
                    if self.if_control_loss:
                        ex_total_loss += ex_loss

                    # extra_pre_label, _, sentence_embedding_two_C, _ = self.BiLSTM_extra(word_embeddings_list, extra_gold_label)
                    # sentence_embedding_two_C = sentence_embedding_two_C.squeeze(0)  # size is 300
                    # if self.valid_flag:
                    #     self.e_pre_labels_list.append(extra_pre_label)
                    #     self.e_gold_labels_list.append(extra_gold_label)

                    sentence_embeddings_list.append(sentence_embedding)

                    # if extra_pre_label == 0:
                    #
                    #     selected_label_embedding = self.label_embedding_list[extra_pre_label, :]
                    #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, extra_pre_label, extra_gold_label)
                    #     sentence_embeddings_list.append(joint_sentence_embedding)
                    #
                    #     # ############################################################
                    #     # # label_embedding_average = torch.mean(self.label_embedding_list[1:7, :], dim=0)  # size is 300
                    #     # # label_embedding_average = self.linear_mask_0_dot_1(label_embedding_average)
                    #     # #
                    #     # # cos_0 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[0, :], dim=0)
                    #     # # cos_1 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[1, :], dim=0)
                    #     # # cos_2 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[2, :], dim=0)
                    #     # # cos_3 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[3, :], dim=0)
                    #     # # cos_4 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[4, :], dim=0)
                    #     # # cos_5 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[5, :], dim=0)
                    #     # # cos_6 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[6, :], dim=0)
                    #     # # cos_average = torch.cosine_similarity(sentence_embedding_two_C, label_embedding_average, dim=0)
                    #     #
                    #     # # if extra_gold_label == 0:
                    #     # #     loss += -cos_0 * 0.5
                    #     # # else:
                    #     # #     loss += -cos_average * 0.5
                    #     # #
                    #     # # if cos_0 > cos_average:
                    #     # # if cos_0 > cos_1 and cos_0 > cos_2 and cos_0 > cos_3 and cos_0 > cos_4 and cos_0 > cos_5 and cos_0 > cos_6:
                    #     # # r = random.randint(1, 100000)
                    #     # if (extra_gold_label == 1 and r < int(0.2 * 100000)) or extra_gold_label == 0:
                    #     #     selected_label_embedding = self.label_embedding_list[extra_pre_label, :]
                    #     #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, extra_pre_label, extra_gold_label)
                    #     #     sentence_embeddings_list.append(joint_sentence_embedding)
                    #     # else:
                    #     #     sentence_embeddings_list.append(sentence_embedding)
                    # else:
                    #     sentence_embeddings_list.append(sentence_embedding)

                if self.mask_p == 0.12:  # choose use sentence_embedding or old_sentence_embedding
                    ex_pre_label, ex_loss, sentence_embedding, _ = self.BiLSTM_ex(word_embeddings_list, ex_gold_label)  # sentence_embedding is 1 * 300
                    sentence_embedding = sentence_embedding.squeeze(0)  # size is 300
                    if self.if_control_loss:
                        ex_total_loss += ex_loss
                    if self.valid_flag:
                        self.e_pre_labels_list.append(ex_pre_label)
                        self.e_gold_labels_list.append(ex_gold_label)

                    extra_pre_label, _, sentence_embedding_two_C, _ = self.BiLSTM_extra(word_embeddings_list, extra_gold_label)
                    sentence_embedding_two_C = sentence_embedding_two_C.squeeze(0)  # size is 300

                    init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
                    independent_BiLSTM_output, _ = self.BiLSTM_1_independent(word_embeddings_list.unsqueeze(0), init_hidden)  # get 1 * sentence_len * 300
                    old_sentence_embedding = torch.max(independent_BiLSTM_output, 1)[0].squeeze(0)  # 300

                    if ex_pre_label == 0 and extra_pre_label != 0:
                        sentence_embeddings_list.append(old_sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)



        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # s.num * 300
        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 307

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, init_hidden)  # 1 * sentence_num * 300
        sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300

        output = self.hidden2tag(sentence_embeddings_output)  # 3 * 7
        output = self.log_softmax(output)  # 3 * 7

        pre_labels_list = []
        for i in range(output.shape[0]):
            pre_labels_list.append(int(torch.argmax(output[i])))

        for i in range(len(gold_labels_list)):
            loss += -output[i][gold_labels_list[i]]

        if self.if_control_loss:
            loss += 0.5 * ex_total_loss

        return pre_labels_list, loss

    def reset(self):
        print("self.reset_num: ", self.reset_num)
        self.reset_num += 1
        print("self.cheat: ", self.cheat)
        print("self.mask_p: ", self.mask_p)
        print("self.valid_flag: ", self.valid_flag)

        if self.valid_flag:
            ex_acc = metrics.accuracy_score(self.e_gold_labels_list, self.e_pre_labels_list)
            print("ex_acc: ", ex_acc)
            print('ex_Confusion Metric: \n', metrics.confusion_matrix(self.e_gold_labels_list, self.e_pre_labels_list))

            used_ex_acc = metrics.accuracy_score(self.used_e_gold_labels_list, self.used_e_pre_label_list)
            print("used_ex_acc: ", used_ex_acc)
            print('used_ex_Confusion Metric: \n', metrics.confusion_matrix(self.used_e_gold_labels_list, self.used_e_pre_label_list))

        self.valid_flag = False

        self.e_pre_labels_list = []
        self.e_gold_labels_list = []

        self.used_e_pre_label_list = []
        self.used_e_gold_labels_list = []

    def cat_and_get_new_embedding(self, sentence_embedding, selected_label_embedding, e_pre_label, e_gold_label):
        joint_sentence_embedding = torch.cat((sentence_embedding, selected_label_embedding), dim=0).unsqueeze(0)  # size is 1 * 600
        joint_sentence_embedding = self.mlp(joint_sentence_embedding).squeeze(0)  # size is 300
        joint_sentence_embedding = self.relu(joint_sentence_embedding)  # size is 300

        if self.valid_flag:
            self.used_e_pre_label_list.append(e_pre_label)
            self.used_e_gold_labels_list.append(e_gold_label)

        return joint_sentence_embedding

    def get_most_similar_memory(self, raw_sentence):
        if '%' in raw_sentence:
            raw_sentence = raw_sentence.replace('%', '%25')
        tokens_list = self.stanford_nlp.word_tokenize(raw_sentence)
        best_sim = 0
        best_ex_pre = 0
        for memory in self.train_data_memory:
            memory_label = memory[1]
            memory_tokens_list = memory[0]
            temp_sim = self.get_tokens_list_sim(tokens_list, memory_tokens_list)
            if temp_sim > best_sim:
                best_sim = temp_sim
                best_ex_pre = memory_label
        return best_ex_pre

    def get_tokens_list_sim(self, list_1, list_memory):
        if list_1 == list_memory:
            return 0
        i = 0
        for t_m in list_memory:
            if t_m in list_1:
                i += 1
        return i / len(list_memory)