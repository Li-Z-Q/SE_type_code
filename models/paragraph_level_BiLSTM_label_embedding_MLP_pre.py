import torch
import gensim
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from stanfordcorenlp import StanfordCoreNLP
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
    def __init__(self, dropout, stanford_path):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.BiLSTM_1 = torch.load('./model_sentence_level_BiLSTM_extra.pt')
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

        self.get_absolut_score = nn.Linear(300, 1)
        self.score_sigmoid = nn.Sigmoid()
        self.score_softmax = nn.Softmax(dim=0)

        self.ex_hidden_2_tag = nn.Linear(300, 7)
        self.ex_softmax = nn.Softmax(dim=0)

        self.hidden2tag = nn.Linear(300, 7)
        self.log_softmax = nn.LogSoftmax()

        self.reset_num = 0
        self.ex_pre_label_list = []
        self.ex_gold_label_list = []

        self.cheat = False
        print("self.cheat: ", self.cheat)

        self.top_percent = 0.5
        print("self.top_percent: ", self.top_percent)

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        old_sentence_embeddings_list = []
        joint_sentence_embeddings_list = []
        score_list = []
        ex_pre_label_list = []
        for i in range(len(sentences_list)):

            sentence = sentences_list[i]
            word_embeddings_list = sentence.unsqueeze(0).cuda()  # 1 * sentence_len * 336

            init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
            word_embeddings_output, _ = self.BiLSTM_1(word_embeddings_list, init_hidden)  # 1 * sentence_len * 300

            sentence_embedding = torch.max(word_embeddings_output[0, :, :], 0)[0]  # size = 300
            old_sentence_embeddings_list.append(sentence_embedding)

            ex_output = self.ex_hidden_2_tag(sentence_embedding)  # size is 7
            ex_output = self.ex_softmax(ex_output)  # size is 7
            ex_pre_label = torch.argmax(ex_output, dim=0)
            ex_pre_label_list.append(ex_pre_label)

            # get joint sentence embedding
            if self.cheat == False:
                average_label_embedding = torch.tensor([0.0 for _ in range(300)]).cuda()
                for j in range(7):
                    average_label_embedding = average_label_embedding + self.label_embedding_list[j, :] * ex_output[j]
            else:
                average_label_embedding = self.label_embedding_list[gold_labels_list[i], :]
            joint_sentence_embedding = torch.cat((sentence_embedding, average_label_embedding), dim=0).unsqueeze(0)  # size is 1 * 600
            joint_sentence_embedding = self.mlp(joint_sentence_embedding).squeeze(0)  # size is 300
            joint_sentence_embedding = self.relu(joint_sentence_embedding)  # size is 300
            joint_sentence_embeddings_list.append(joint_sentence_embedding)

            # get absolute score and loss of this jooinft sentence embedding
            score = self.get_absolut_score(joint_sentence_embedding)  # size is 1
            score = self.score_sigmoid(score)
            score_list.append(score)

        score_list = torch.stack(score_list)  # s.num * 1
        score_list = score_list.resize(1, score_list.shape[0])  # 1 * s.num
        score_list = score_list.squeeze(0)
        score_list = self.score_softmax(score_list)  # size is s.num
        score_index_list = torch.sort(score_list, dim=0, descending=True)[1]
        score_index_list = score_index_list[:int(self.top_percent*len(score_index_list))]  # use top 20%

        # get sentences_embedding list from old or joint
        sentence_embeddings_list = []
        for i in range(len(old_sentence_embeddings_list)):
            if i in score_index_list:
                sentence_embeddings_list.append(joint_sentence_embeddings_list[i])
            else:
                sentence_embeddings_list.append(old_sentence_embeddings_list[i])

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

        for i in range(len(gold_labels_list)):
            if gold_labels_list[i] != ex_pre_label_list[i] and i in score_index_list:
                loss += (score_list[i] * 1.5)
            if gold_labels_list[i] == ex_pre_label_list[i] and i not in score_index_list:
                loss += -(score_list[i] * 1.5)

        for i in range(len(score_index_list)):
            self.ex_pre_label_list.append(int(ex_pre_label_list[i].cpu()))
            self.ex_gold_label_list.append(gold_labels_list[i])

        return pre_labels_list, loss

    def reset(self):
        print("self.reset_num: ", self.reset_num)
        self.reset_num += 1
        print("self.cheat: ", self.cheat)
        # print(self.ex_pre_label_list)
        # print(self.ex_gold_label_list)
        if self.reset_num > 1:
            ex_acc = metrics.accuracy_score(self.ex_gold_label_list, self.ex_pre_label_list)
            print("ex_acc: ", ex_acc)
        self.ex_pre_label_list = []
        self.ex_gold_label_list = []

