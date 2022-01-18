import random

import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
from models.sentence_level_BERT_extra import MyModel as MyModel_ex
from transformers import BertTokenizer, BertConfig, BertModel, BertForTokenClassification


print("paragraph level BERT label embedding MLP pre")

tokenizer = BertTokenizer.from_pretrained('pre_train')
model_config = BertConfig.from_pretrained('pre_train')
model_config.num_labels = 7
# model_config.output_attentions = True
# model_config.output_hidden_states = True


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


class MyModel(nn.Module):
    def __init__(self, dropout, stanford_path, pre_model_path, cheat, mask_p):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        tmp = MyModel_ex()
        self.bert_model_1 = tmp.load_model(path=pre_model_path)
        self.BERT_1_r = True
        print("self.BERT_1_require_grad: ", self.BERT_1_r)
        for p in self.bert_model_1.parameters():
            p.requires_grad = self.BERT_1_r

        self.bert_model_2 = BertForTokenClassification.from_pretrained('pre_train/', config=model_config)

        self.label_embedding_list = nn.Parameter(do_label_embedding(), requires_grad=True)

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

        self.reliability_list = []
        self.c_reliability_list = []
        self.w_reliability_list = []

        self.valid_flag = False

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        old_sentence_embeddings_list = []
        joint_sentence_embeddings_list = []
        ex_pre_label_list = []
        for i in range(len(sentences_list)):

            # get sentence embedding
            sentence = sentences_list[i]
            word_embeddings_list = sentence.cuda()  # sentence_len * 300

            ex_pre_label, _, sentence_embedding, softmax_output = self.BiLSTM_1(word_embeddings_list, 0)  # 1 * 300
            sentence_embedding = sentence_embedding.squeeze(0)  # size is 300

            ex_pre_label_list.append(ex_pre_label)
            reliability = softmax_output[ex_pre_label]
            self.reliability_list.append(float(reliability))

            if self.valid_flag:
                if ex_pre_label == gold_labels_list[i]:
                    self.c_reliability_list.append(float(reliability))
                else:
                    self.w_reliability_list.append(float(reliability))

            old_sentence_embeddings_list.append(sentence_embedding)

            # get joint sentence embedding
            average_label_embedding = None
            if self.cheat == "False":
                average_label_embedding = self.label_embedding_list[ex_pre_label, :]
            else:
                if self.mask_p >= 0:  # use masked gold label
                    save_p = 1 - self.mask_p
                    r = random.randint(1, 100000)
                    if r < int(save_p*100000):
                        average_label_embedding = self.label_embedding_list[gold_labels_list[i], :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]
                if self.mask_p == -1:  # if ex_pre_label is correct, then cat it
                    if ex_pre_label == gold_labels_list[i]:
                        average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]
                if self.mask_p == -2:  # if ex_pre_label is wrong, then cat it
                    if ex_pre_label != gold_labels_list[i]:
                        average_label_embedding = self.label_embedding_list[ex_pre_label, :]
                    else:
                        random_gold = random.randint(0, 6)
                        average_label_embedding = self.label_embedding_list[random_gold, :]

            joint_sentence_embedding = torch.cat((sentence_embedding, average_label_embedding), dim=0).unsqueeze(0)  # size is 1 * 600
            joint_sentence_embedding = self.mlp(joint_sentence_embedding).squeeze(0)  # size is 300
            joint_sentence_embedding = self.relu(joint_sentence_embedding)  # size is 300
            joint_sentence_embeddings_list.append(joint_sentence_embedding)

        # get sentences_embedding list from old or joint
        sentence_embeddings_list = []
        for i in range(len(old_sentence_embeddings_list)):
            sentence_embeddings_list.append(joint_sentence_embeddings_list[i])
            # sentence_embeddings_list.append(old_sentence_embeddings_list[i])
            if self.valid_flag:
                self.ex_pre_label_list.append(ex_pre_label_list[i])
                self.gold_labels_list.append(gold_labels_list[i])

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

            reliability_np = np.array(self.reliability_list)
            print("np.mean(reliability_np): ", np.mean(reliability_np))
            print("np.median(reliability_np): ", np.median(reliability_np))

            c_reliability_np = np.array(self.c_reliability_list)
            print("np.mean(c_reliability_np): ", np.mean(c_reliability_np))
            print("np.median(c_reliability_np): ", np.median(c_reliability_np))

            w_reliability_np = np.array(self.w_reliability_list)
            print("np.mean(w_reliability_np): ", np.mean(w_reliability_np))
            print("np.median(w_reliability_np): ", np.median(w_reliability_np))

        self.valid_flag = False

        self.ex_pre_label_list = []
        self.gold_labels_list = []

        self.reliability_list = []
        self.c_reliability_list = []
        self.w_reliability_list = []