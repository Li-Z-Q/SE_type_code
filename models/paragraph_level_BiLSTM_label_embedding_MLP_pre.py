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
        # tmp = MyModel_ex()
        # self.BiLSTM_ex = tmp.load_model(path=ex_model_path)

        self.if_control_loss = bool(if_control_loss)
        self.if_use_memory = bool(if_use_memory)
        print("\nself.if_control_loss: ", self.if_control_loss)
        print("self.if_use_memory: ", self.if_use_memory)

        self.cheat = cheat
        self.mask_p = mask_p
        print("self.mask_p: ", self.mask_p)
        print("self.cheat: ", self.cheat)

        self.train_data_memory = train_data_memory

        self.stanford_nlp = StanfordCoreNLP(stanford_path)

        if self.if_use_memory == False:  # use ex_model do pre_label
            self.BiLSTM_ex = ex_model
            print("bool(ex_model_grad): ", bool(ex_model_grad))
            for p in self.BiLSTM_ex.parameters():
                p.requires_grad = bool(ex_model_grad)

        self.BiLSTM_ex_extra = ex_model_extra
        for p in self.BiLSTM_ex_extra.parameters():
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

        self.ex_pre_labels_list = []
        self.ex_gold_labels_list = []

        self.used_ex_pre_label_list = []
        self.used_ex_gold_labels_list = []

        self.reliability_list = []
        self.c_reliability_list = []
        self.w_reliability_list = []

        self.valid_flag = False
        self.threshold = 0.8
        print("self.threshold: ", self.threshold)

    def forward(self, sentences_list_with_without_raw, gold_labels_list):  # [4*3336, 7*336, 1*336]
        # old_sentence_embeddings_list = []
        # joint_sentence_embeddings_list = []
        # ex_pre_label_list = []

        sentences_list = sentences_list_with_without_raw[0]
        raw_sentences_list = sentences_list_with_without_raw[1]

        loss = 0
        ex_total_loss = 0
        reliability_list = []
        ex_pre_labels_list = []
        ex_gold_labels_list = []
        sentence_embeddings_list = []
        for i in range(len(sentences_list)):

            # get sentence embedding
            sentence = sentences_list[i]
            word_embeddings_list = sentence.cuda()  # sentence_len * 300

            if self.mask_p == 0.1 or self.mask_p == 0.2 or self.mask_p == 0.15:  # only 2 class for ex_model
                ex_gold_label = int(gold_labels_list[i] != 0)
            else:
                ex_gold_label = gold_labels_list[i]

            if self.if_use_memory:
                init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
                independent_BiLSTM_output, _ = self.BiLSTM_1_independent(word_embeddings_list.unsqueeze(0), init_hidden)  # get 1 * sentence_len * 300
                sentence_embedding = torch.max(independent_BiLSTM_output, 1)[0]  # 1 * 300
                raw_sentence = raw_sentences_list[i]  # is a text: "i love you"
                ex_pre_label = self.get_most_similar_memory(raw_sentence)
                reliability = 0

            else:  # use ex_pre_label
                if self.mask_p == 0.1 or self.mask_p == 0.2 or self.mask_p == 0.12:  # use BiLSTM_ex -> BiLSTM_2, and BiLSTM_ex_extra for ex_pre_label
                    _, ex_loss, sentence_embedding, softmax_output = self.BiLSTM_ex(word_embeddings_list, ex_gold_label) # sentence_embedding is 1 * 300
                    ex_pre_label, _, sentence_embedding_two_C, _ = self.BiLSTM_ex_extra(word_embeddings_list, ex_gold_label)
                    sentence_embedding_two_C = sentence_embedding_two_C.squeeze(0)  # size is 300
                    # if ex_pre_label != ex_gold_label:
                    #     ex_pre_label = 1
                    reliability = softmax_output[ex_pre_label]
                    reliability_list.append(float(reliability))
                    self.reliability_list.append(float(reliability))
                    if self.if_control_loss:
                        ex_total_loss += ex_loss

                elif self.mask_p == 0.15:
                    ex_pre_label, _, sentence_embedding_two_C, _ = self.BiLSTM_ex_extra(word_embeddings_list, ex_gold_label)
                    sentence_embedding_two_C = sentence_embedding_two_C.squeeze(0)  # size is 300
                    init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
                    independent_BiLSTM_output, _ = self.BiLSTM_1_independent(word_embeddings_list.unsqueeze(0), init_hidden)  # get 1 * sentence_len * 300
                    sentence_embedding = torch.max(independent_BiLSTM_output, 1)[0]  # 1 * 300
                    reliability = 0

                else:  # the first layer bilstm as ex_pre_label provider and embedding
                    ex_pre_label, ex_loss, sentence_embedding, softmax_output = self.BiLSTM_ex(word_embeddings_list, ex_gold_label)  # 1 * 300
                    reliability = softmax_output[ex_pre_label]
                    reliability_list.append(float(reliability))
                    self.reliability_list.append(float(reliability))
                    if self.if_control_loss:
                        ex_total_loss += ex_loss

            if self.valid_flag:
                self.ex_pre_labels_list.append(ex_pre_label)
                self.ex_gold_labels_list.append(ex_gold_label)

                if ex_pre_label == ex_gold_label:
                    self.c_reliability_list.append(float(reliability))
                else:
                    self.w_reliability_list.append(float(reliability))

            sentence_embedding = sentence_embedding.squeeze(0)  # size is 300

            if self.cheat == "False":
                if self.mask_p == 0.0:  # use all pre_label
                    selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                    sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p == 0.1:  # use ex_pre_label == 0:, only have two class
                    if ex_pre_label == 0:  # and ex_gold_label == 0:

                        selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                        sentence_embeddings_list.append(joint_sentence_embedding)

                        # ############################################################
                        # label_embedding_average = torch.mean(self.label_embedding_list[1:7, :], dim=0)  # size is 300
                        # label_embedding_average = self.linear_mask_0_dot_1(label_embedding_average)
                        # #
                        # cos_0 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[0, :], dim=0)
                        # # cos_1 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[1, :], dim=0)
                        # # cos_2 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[2, :], dim=0)
                        # # cos_3 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[3, :], dim=0)
                        # # cos_4 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[4, :], dim=0)
                        # # cos_5 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[5, :], dim=0)
                        # # cos_6 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[6, :], dim=0)
                        # cos_average = torch.cosine_similarity(sentence_embedding_two_C, label_embedding_average, dim=0)
                        #
                        # # if ex_gold_label == 0:
                        # #     loss += -cos_0 * 0.5
                        # # else:
                        # #     loss += -cos_average * 0.5
                        # #
                        # # cos_0 *= 0.4
                        # if cos_0 > cos_average + 10:
                        # # if cos_0 > cos_1 and cos_0 > cos_2 and cos_0 > cos_3 and cos_0 > cos_4 and cos_0 > cos_5 and cos_0 > cos_6:
                        #     selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                        #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                        #     sentence_embeddings_list.append(joint_sentence_embedding)
                        # else:
                        #     sentence_embeddings_list.append(sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)
                if self.mask_p == 0.12:  # only have two class
                    ex_pre_labels_list.append(ex_pre_label)
                    ex_gold_labels_list.append(ex_gold_label)
                    sentence_embeddings_list.append(sentence_embedding)

                if self.mask_p == 0.15:
                    if ex_pre_label == 0:  # and ex_gold_label == 0:

                        # selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                        # joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                        # sentence_embeddings_list.append(joint_sentence_embedding)

                        ############################################################
                        label_embedding_average = torch.mean(self.label_embedding_list[1:7, :], dim=0)  # size is 300
                        label_embedding_average = self.linear_mask_0_dot_1(label_embedding_average)
                        #
                        cos_0 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[0, :],
                                                        dim=0)
                        # cos_1 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[1, :], dim=0)
                        # cos_2 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[2, :], dim=0)
                        # cos_3 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[3, :], dim=0)
                        # cos_4 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[4, :], dim=0)
                        # cos_5 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[5, :], dim=0)
                        # cos_6 = torch.cosine_similarity(sentence_embedding_two_C, self.label_embedding_list[6, :], dim=0)
                        cos_average = torch.cosine_similarity(sentence_embedding_two_C, label_embedding_average, dim=0)

                        if ex_gold_label == 0:
                            loss += -cos_0 * 0.5
                        else:
                            loss += -cos_average * 0.5
                        #
                        # cos_0 *= 0.4
                        if cos_0 > cos_average + 10:
                            # if cos_0 > cos_1 and cos_0 > cos_2 and cos_0 > cos_3 and cos_0 > cos_4 and cos_0 > cos_5 and cos_0 > cos_6:
                            selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                            joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding,
                                                                                      selected_label_embedding,
                                                                                      ex_pre_label, ex_gold_label)
                            sentence_embeddings_list.append(joint_sentence_embedding)
                        else:
                            sentence_embeddings_list.append(sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)

                if self.mask_p == 0.2:  # use ex_pre_label == 0:, only have two class
                    if ex_pre_label != 0:  # and ex_gold_label != 0:

                        selected_label_embedding = self.label_embedding_list[0, :]
                        # selected_label_embedding = torch.mean(self.label_embedding_list[1:7, :], dim=0)  # size is 300
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                        sentence_embeddings_list.append(joint_sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)

                # if self.mask_p == 1.0:  # only cat correct_pre, others use un-cat
                #     if ex_pre_label == ex_gold_label:
                #         selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                #         joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                #         sentence_embeddings_list.append(joint_sentence_embedding)
                #     else:
                #         sentence_embeddings_list.append(sentence_embedding)
                if self.mask_p == 2.0:  # only cat up_threshold__pre, others use un-cat
                    if reliability_list[i] > self.threshold:
                        selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                        sentence_embeddings_list.append(joint_sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)
                # if self.mask_p == 3.0:  # only cat up_threshold__pre, others cat random
                #     if float(reliability) > self.threshold:
                #         selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                #     else:
                #         random_gold = random.randint(0, 6)
                #         selected_label_embedding = self.label_embedding_list[random_gold, :]
                #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                #     sentence_embeddings_list.append(joint_sentence_embedding)
                # if self.mask_p == 4.0:  # cat all pre_label's random_reverse
                #     while True:
                #         random_pre = random.randint(0, 6)
                #         if random_pre != ex_pre_label:
                #             break
                #     selected_label_embedding = self.label_embedding_list[random_pre, :]
                #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                #     sentence_embeddings_list.append(joint_sentence_embedding)

            else:  # cheat == True
                if self.mask_p >= 0:  # use masked gold label
                    save_p = 1 - self.mask_p
                    r = random.randint(1, 100000)
                    if r < int(save_p*100000):
                        selected_label_embedding = self.label_embedding_list[ex_gold_label, :]
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                        sentence_embeddings_list.append(joint_sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)
                # if self.mask_p == -1:  # if ex_pre_label is correct, then cat it, else random
                #     if ex_pre_label == ex_gold_label:
                #         selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                #     else:
                #         random_gold = random.randint(0, 6)
                #         selected_label_embedding = self.label_embedding_list[random_gold, :]
                #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                #     sentence_embeddings_list.append(joint_sentence_embedding)
                # if self.mask_p == -2:  # if ex_pre_label is wrong, then cat ex_pre_label, else random
                #     if ex_pre_label != ex_gold_label:
                #         selected_label_embedding = self.label_embedding_list[ex_pre_label, :]
                #     else:
                #         random_gold = random.randint(0, 6)
                #         selected_label_embedding = self.label_embedding_list[random_gold, :]
                #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                #     sentence_embeddings_list.append(joint_sentence_embedding)
                # if self.mask_p == -2.5:  # if ex_pre_label is wrong, then cat it, else random
                #     if ex_pre_label != ex_gold_label:
                #         selected_label_embedding = self.label_embedding_list[ex_gold_label, :]
                #     else:
                #         random_gold = random.randint(0, 6)
                #         selected_label_embedding = self.label_embedding_list[random_gold, :]
                #     joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                #     sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p < -10:  # if golden label is someone, cat it, else use un-cat
                    someone = -(self.mask_p + 100)  # if self.mask_p = -101, someone = 1
                    if ex_gold_label == someone:
                        selected_label_embedding = self.label_embedding_list[ex_gold_label, :]
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label)
                        sentence_embeddings_list.append(joint_sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)

        if self.mask_p == 0.12:
            for i in range(len(sentence_embeddings_list)):
                for j in range(len(sentence_embeddings_list)):
                    if ex_pre_labels_list[j] == ex_pre_labels_list[i]:
                        sentence_embeddings_list[i] = self.cat_and_get_new_embedding(sentence_embeddings_list[i], sentence_embeddings_list[i], 0, 0)

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
            ex_acc = metrics.accuracy_score(self.ex_gold_labels_list, self.ex_pre_labels_list)
            print("ex_acc: ", ex_acc)
            print('ex_Confusion Metric: \n', metrics.confusion_matrix(self.ex_gold_labels_list, self.ex_pre_labels_list))

            used_ex_acc = metrics.accuracy_score(self.used_ex_gold_labels_list, self.used_ex_pre_label_list)
            print("used_ex_acc: ", used_ex_acc)
            print('used_ex_Confusion Metric: \n', metrics.confusion_matrix(self.used_ex_gold_labels_list, self.used_ex_pre_label_list))

            reliability_np = np.array(self.reliability_list)
            print("np.mean(reliability_np): ", np.mean(reliability_np))
            print("np.median(reliability_np): ", np.median(reliability_np))

            c_reliability_np = np.array(self.c_reliability_list)
            print("np.mean(c_reliability_np): ", np.mean(c_reliability_np))
            print("np.median(c_reliability_np): ", np.median(c_reliability_np))

            w_reliability_np = np.array(self.w_reliability_list)
            print("np.mean(w_reliability_np): ", np.mean(w_reliability_np))
            print("np.median(w_reliability_np): ", np.median(w_reliability_np))

            self.threshold = np.median(c_reliability_np)
            print("self.threshold: ", self.threshold)

        self.valid_flag = False

        self.ex_pre_labels_list = []
        self.ex_gold_labels_list = []

        self.used_ex_pre_label_list = []
        self.used_ex_gold_labels_list = []

        self.reliability_list = []
        self.c_reliability_list = []
        self.w_reliability_list = []

    def cat_and_get_new_embedding(self, sentence_embedding, selected_label_embedding, ex_pre_label, ex_gold_label):
        joint_sentence_embedding = torch.cat((sentence_embedding, selected_label_embedding), dim=0).unsqueeze(0)  # size is 1 * 600
        joint_sentence_embedding = self.mlp(joint_sentence_embedding).squeeze(0)  # size is 300
        joint_sentence_embedding = self.relu(joint_sentence_embedding)  # size is 300

        if self.valid_flag:
            self.used_ex_pre_label_list.append(ex_pre_label)
            self.used_ex_gold_labels_list.append(ex_gold_label)

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