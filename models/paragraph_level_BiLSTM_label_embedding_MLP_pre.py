import random

import torch
import gensim
import numpy as np
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from stanfordcorenlp import StanfordCoreNLP
from models.sentence_level_BiLSTM_extra import MyModel as MyModel_ex
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list

print("paragraph level BiLSTM label embedding MLP pre")


def do_label_embedding(stanford_path):
    word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('resource/GoogleNews-vectors-negative300.bin', binary=True)
    seType_dict = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']

    stanford_nlp = StanfordCoreNLP(stanford_path)

    label_embedding_list = []
    for label in seType_dict:
        word_embeddings_list = from_sentence_2_word_embeddings_list(label, stanford_nlp, word2vec_vocab).cuda()
        label_embedding = torch.max(word_embeddings_list, dim=0)[0]  # will get size is 300
        label_embedding_list.append(label_embedding)
    label_embedding_list = torch.stack(label_embedding_list)  # 8 * 300

    stanford_nlp.close()

    return label_embedding_list


def do_label_embedding_random():
    label_embedding_list = []
    for i in range(7):
        label_embedding = torch.randn(300)
        label_embedding_list.append(label_embedding)
    label_embedding_list = torch.stack(label_embedding_list)  # 8 * 300
    return label_embedding_list


class MyModel(nn.Module):
    def __init__(self, dropout, stanford_path, pre_model, cheat, mask_p, bilstm_1_grad, if_control_loss, if_use_independent, if_use_memory):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # tmp = MyModel_ex()
        # self.BiLSTM_1 = tmp.load_model(path=pre_model_path)

        self.if_control_loss = bool(if_control_loss)
        self.if_use_independent = bool(if_use_independent)
        self.if_use_memory = bool(if_use_memory)
        print("self.if_control_loss: ", self.if_control_loss)
        print("self.if_use_memory: ", self.if_use_memory)
        print("self.if_use_independent: ", self.if_use_independent)

        self.BiLSTM_1 = pre_model
        self.BiLSTM_1_r = bool(bilstm_1_grad)
        print("self.BiLSTM_1_require_grad: ", self.BiLSTM_1_r)
        for p in self.BiLSTM_1.parameters():
            p.requires_grad = self.BiLSTM_1_r

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

        self.label_embedding_list = nn.Parameter(do_label_embedding(stanford_path), requires_grad=True)

        self.mlp = nn.Linear(in_features=600, out_features=300)
        self.relu = nn.ReLU()

        self.hidden2tag = nn.Linear(300, 7)
        self.log_softmax = nn.LogSoftmax()

        self.reset_num = 0
        self.cheat = cheat
        self.mask_p = mask_p
        print("self.cheat: ", self.cheat)
        print("self.mask_p: ", self.mask_p)

        self.ex_pre_label_list = []
        self.gold_labels_list = []

        self.used_ex_pre_label_list = []
        self.used_gold_labels_list = []

        self.reliability_list = []
        self.c_reliability_list = []
        self.w_reliability_list = []

        self.valid_flag = False
        self.threshold = 0.8
        print("self.threshold: ", self.threshold)

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        # old_sentence_embeddings_list = []
        # joint_sentence_embeddings_list = []
        # ex_pre_label_list = []
        ex_total_loss = 0
        reliability_list = []
        sentence_embeddings_list = []
        for i in range(len(sentences_list)):

            # get sentence embedding
            sentence = sentences_list[i]
            word_embeddings_list = sentence.cuda()  # sentence_len * 300

            ex_pre_label, ex_loss, sentence_embedding, softmax_output = self.BiLSTM_1(word_embeddings_list, gold_labels_list[i])  # 1 * 300
            if self.if_control_loss:
                ex_total_loss += ex_loss

            if self.if_use_independent:  # use extra bilstm to provide pre_label for original 2 layer bilstm
                # print("self.if_use_independent: "self.if_use_independent)
                init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
                independent_BiLSTM_output, _ = self.BiLSTM_1_independent(word_embeddings_list.unsqueeze(0), init_hidden)  # get 1 * sentence_len * 300
                sentence_embedding = torch.max(independent_BiLSTM_output, 1)[0]  # 1 * 300

            if self.if_use_memory:
                init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
                independent_BiLSTM_output, _ = self.BiLSTM_1_independent(word_embeddings_list.unsqueeze(0), init_hidden)  # get 1 * sentence_len * 300
                sentence_embedding = torch.max(independent_BiLSTM_output, 1)[0]  # 1 * 300

            sentence_embedding = sentence_embedding.squeeze(0)  # size is 300

            reliability = softmax_output[ex_pre_label]
            reliability_list.append(float(reliability))
            self.reliability_list.append(float(reliability))

            if self.valid_flag:
                self.ex_pre_label_list.append(ex_pre_label)
                self.gold_labels_list.append(gold_labels_list[i])

                if ex_pre_label == gold_labels_list[i]:
                    self.c_reliability_list.append(float(reliability))
                else:
                    self.w_reliability_list.append(float(reliability))

            if self.cheat == "False":
                if self.mask_p == 0.0:  # use all pre_label
                    average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                    sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p == 1.0:  # only cat correct_pre, others use un-cat
                    if ex_pre_label == gold_labels_list[i]:
                        average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                        sentence_embeddings_list.append(joint_sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)
                if self.mask_p == 2.0:  # only cat up_threshold__pre, others use un-cat
                    if reliability_list[i] > self.threshold:
                        average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                        sentence_embeddings_list.append(joint_sentence_embedding)
                    else:
                        sentence_embeddings_list.append(sentence_embedding)
                if self.mask_p == 3.0:  # only cat up_threshold__pre, others cat random
                    if float(reliability) > self.threshold:
                        average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                    sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p == 4.0:  # cat all pre_label's random_reverse
                    while True:
                        random_pre = random.randint(0, 6)
                        if random_pre != ex_pre_label:
                            break
                    average_label_embedding = self.label_embedding_list[random_pre, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                    sentence_embeddings_list.append(joint_sentence_embedding)

            else:  # cheat == True
                if self.mask_p >= 0:  # use masked gold label
                    save_p = 1 - self.mask_p
                    r = random.randint(1, 100000)
                    if r < int(save_p*100000):
                        average_label_embedding = self.label_embedding_list[gold_labels_list[i], :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                    sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p == -1:  # if ex_pre_label is correct, then cat it, else random
                    if ex_pre_label == gold_labels_list[i]:
                        average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                    sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p == -2:  # if ex_pre_label is wrong, then cat ex_pre_label, else random
                    if ex_pre_label != gold_labels_list[i]:
                        average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                    sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p == -2.5:  # if ex_pre_label is wrong, then cat it, else random
                    if ex_pre_label != gold_labels_list[i]:
                        average_label_embedding = self.label_embedding_list[gold_labels_list[i], :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]
                    joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                    sentence_embeddings_list.append(joint_sentence_embedding)
                if self.mask_p == -3:  # if golden label is 0, cat it, else use un-cat
                    if gold_labels_list[i] == 0:
                        average_label_embedding = self.label_embedding_list[gold_labels_list[i], :]
                        joint_sentence_embedding = self.cat_and_get_new_embedding(sentence_embedding, average_label_embedding, ex_pre_label, gold_labels_list[i])
                        sentence_embeddings_list.append(joint_sentence_embedding)
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

        loss = 0
        for i in range(len(gold_labels_list)):
            label = gold_labels_list[i]
            loss += -output[i][label]

        if self.if_control_loss:
            loss += 0.5 * ex_total_loss

        return pre_labels_list, loss

    def reset(self):
        print("self.reset_num: ", self.reset_num)
        self.reset_num += 1
        print("self.cheat: ", self.cheat)
        print("self.mask_p: ", self.mask_p)
        print("self.valid_flag: ", self.valid_flag)
        print("self.BiLSTM_1_require_grad: ", self.BiLSTM_1_r)

        if self.valid_flag:
            ex_acc = metrics.accuracy_score(self.gold_labels_list, self.ex_pre_label_list)
            print("ex_acc: ", ex_acc)
            print('ex_Confusion Metric: \n', metrics.confusion_matrix(self.gold_labels_list, self.ex_pre_label_list))

            used_ex_acc = metrics.accuracy_score(self.used_gold_labels_list, self.used_ex_pre_label_list)
            print("used_ex_acc: ", used_ex_acc)
            print('used_ex_Confusion Metric: \n', metrics.confusion_matrix(self.used_gold_labels_list, self.used_ex_pre_label_list))

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

        self.ex_pre_label_list = []
        self.gold_labels_list = []

        self.used_ex_pre_label_list = []
        self.used_gold_labels_list = []

        self.reliability_list = []
        self.c_reliability_list = []
        self.w_reliability_list = []

    def cat_and_get_new_embedding(self, sentence_embedding, average_label_embedding, ex_pre_label, gold_label):
        joint_sentence_embedding = torch.cat((sentence_embedding, average_label_embedding), dim=0).unsqueeze(0)  # size is 1 * 600
        joint_sentence_embedding = self.mlp(joint_sentence_embedding).squeeze(0)  # size is 300
        joint_sentence_embedding = self.relu(joint_sentence_embedding)  # size is 300

        if self.valid_flag:
            self.used_ex_pre_label_list.append(ex_pre_label)
            self.used_gold_labels_list.append(gold_label)

        return joint_sentence_embedding