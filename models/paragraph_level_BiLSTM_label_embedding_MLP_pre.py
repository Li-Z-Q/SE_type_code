import torch
import gensim
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from stanfordcorenlp import StanfordCoreNLP
from models.sentence_level_BiLSTM_extra import MyModel as MyModel_ex
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list

print("paragraph level BiLSTM label embedding MLP pre")


def do_label_embedding(stanford_path):
    word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('resource/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
    seType_dict = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']

    stanford_nlp = StanfordCoreNLP(stanford_path)

    label_embedding_list = []

    for label in seType_dict:
        # print("label: ", label)
        word_embeddings_list = from_sentence_2_word_embeddings_list(label, stanford_nlp, word2vec_vocab).cuda()
        # print("word_embeddings_list.shape: ", word_embeddings_list.shape)
        label_embedding = torch.max(word_embeddings_list, dim=0)[0]  # will get size is 300
        # print("label_embedding.shape: ", label_embedding.shape)
        label_embedding_list.append(label_embedding)
        # input()

    # average_labels_embedding = torch.mean(torch.stack(label_embedding_list), dim=0)  # will get size is 300
    # label_embedding_list.append(average_labels_embedding)

    stanford_nlp.close()

    label_embedding_list = torch.stack(label_embedding_list)  # 8 * 300

    return label_embedding_list


class MyModel(nn.Module):
    def __init__(self, dropout, stanford_path, pre_model_path):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        tmp = MyModel_ex()
        self.BiLSTM_1 = tmp.load_model(path=pre_model_path)
        for p in self.BiLSTM_1.parameters():
            p.requires_grad = False

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
        self.cheat = False
        print("self.cheat: ", self.cheat)

        self.ex_pre_label_list = []
        self.gold_labels_list = []

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        old_sentence_embeddings_list = []
        joint_sentence_embeddings_list = []
        ex_pre_label_list = []
        for i in range(len(sentences_list)):

            # get sentence embedding
            sentence = sentences_list[i]
            word_embeddings_list = sentence.cuda()  # sentence_len * 300

            ex_pre_label, _, sentence_embedding = self.BiLSTM_1(word_embeddings_list, 0)  # 1 * 300
            sentence_embedding = sentence_embedding.squeeze(0)  # size is 300

            ex_pre_label_list.append(ex_pre_label)
            old_sentence_embeddings_list.append(sentence_embedding)

            # get joint sentence embedding
            if self.cheat == False:
                average_label_embedding = self.label_embedding_list[ex_pre_label, :]
            else:
                average_label_embedding = self.label_embedding_list[gold_labels_list[i], :]
            joint_sentence_embedding = torch.cat((sentence_embedding, average_label_embedding), dim=0).unsqueeze(0)  # size is 1 * 600
            joint_sentence_embedding = self.mlp(joint_sentence_embedding).squeeze(0)  # size is 300
            joint_sentence_embedding = self.relu(joint_sentence_embedding)  # size is 300
            joint_sentence_embeddings_list.append(joint_sentence_embedding)

        # get sentences_embedding list from old or joint
        sentence_embeddings_list = []
        for i in range(len(old_sentence_embeddings_list)):
            sentence_embeddings_list.append(joint_sentence_embeddings_list[i])
            # sentence_embeddings_list.append(old_sentence_embeddings_list[i])

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

        self.ex_pre_label_list = ex_pre_label_list
        self.gold_labels_list = gold_labels_list

        return pre_labels_list, loss

    def reset(self):
        print("self.reset_num: ", self.reset_num)
        self.reset_num += 1
        print("self.cheat: ", self.cheat)
        if self.reset_num > 1:
            ex_acc = metrics.accuracy_score(self.gold_labels_list, self.ex_pre_label_list)
            print("ex_acc: ", ex_acc)

