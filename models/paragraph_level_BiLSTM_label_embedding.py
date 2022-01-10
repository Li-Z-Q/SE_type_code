import torch
import gensim
import torch.nn as nn
from torch.autograd import Variable
from stanfordcorenlp import StanfordCoreNLP
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list


print("paragraph level BiLSTM label embedding")


def do_label_embedding(stanford_path):
    word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('resource/GoogleNews-vectors-negative300.bin', binary=True)
    seType_dict = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']

    stanford_nlp = StanfordCoreNLP(stanford_path)

    label_embedding_list = []

    for label in seType_dict:
        word_embeddings_list = from_sentence_2_word_embeddings_list(label, stanford_nlp, word2vec_vocab).cuda()
        word_embeddings_list = torch.max(word_embeddings_list, dim=1)[0]   # will get size is 300
        label_embedding_list.append(word_embeddings_list)

    average_labels_embedding = torch.mean(torch.stack(label_embedding_list), dim=0)  # will get size is 300
    label_embedding_list.append(average_labels_embedding)

    stanford_nlp.close()
    return label_embedding_list


class MyModel(nn.Module):
    def __init__(self, dropout, stanford_path):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.BiLSTM_1 = nn.LSTM(300,
                                300 // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)
        self.BiLSTM_2 = nn.LSTM(600,
                                300 // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)

        self.label_embedding_list = do_label_embedding(stanford_path)
        self.ex_hidden_2_tag = nn.Linear(300, 7)
        self.ex_softmax = nn.Softmax(dim=0)

        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

        self.gold_prob = 0
        self.gold_num = 1

    def forward(self, sentences_list, gold_labels_list):  # [4*3336, 7*336, 1*336]
        sentence_embeddings_list = []
        for i in range(len(sentences_list)):

            sentence = sentences_list[i]
            word_embeddings_list = sentence.unsqueeze(0).cuda()  # 1 * sentence_len * 336

            init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
            word_embeddings_output, _ = self.BiLSTM_1(word_embeddings_list, init_hidden)  # 1 * sentence_len * 300

            sentence_embedding = torch.max(word_embeddings_output[0, :, :], 0)[0]  # size = 300

            gold_label = gold_labels_list[i]
            # one_hot_vector = [0 for _ in range(7)]
            # one_hot_vector[gold_label] = 1
            ex_output = self.ex_hidden_2_tag(sentence_embedding)  # size is 7
            ex_output = self.ex_softmax(ex_output)  # size is 7
            temp_pre_label = torch.argmax(ex_output, dim=0)
            if temp_pre_label == gold_label:
                # print(ex_output, temp_pre_label)
                self.gold_prob += ex_output[temp_pre_label]
                self.gold_num += 1
            if ex_output[temp_pre_label] > 0.165:
                sentence_embedding = torch.cat((sentence_embedding, self.label_embedding_list[temp_pre_label]), dim=0).cuda()  # size is 300 + 300
            else:
                sentence_embedding = torch.cat((sentence_embedding, self.label_embedding_list[7]), dim=0).cuda()

            sentence_embeddings_list.append(sentence_embedding)
        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 307
        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 307

        init_hidden = (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda())
        sentence_embeddings_output, _ = self.BiLSTM_2(sentence_embeddings_list, init_hidden)  # 1 * sentence_num * 300
        sentence_embeddings_output = sentence_embeddings_output.squeeze(0)  # sentence_num * 300

        output = self.hidden2tag(sentence_embeddings_output)  # 3 * 7

        output = self.softmax(output)  # 3 * 7

        pre_labels_list = []
        for i in range(output.shape[0]):
            pre_labels_list.append(int(torch.argmax(output[i])))

        loss = 0
        for i in range(len(gold_labels_list)):
            label = gold_labels_list[i]
            loss += -output[i][label]

        return pre_labels_list, loss

    def reset(self):
        print(self.gold_prob / self.gold_num)
        self.gold_prob = 0
        self.gold_num = 1


