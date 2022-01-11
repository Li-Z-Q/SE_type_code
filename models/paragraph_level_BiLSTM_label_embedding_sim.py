import torch
import gensim
import torch.nn as nn
from torch.autograd import Variable
from stanfordcorenlp import StanfordCoreNLP
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list


print("paragraph level BiLSTM label embedding sim")


def do_label_embedding(stanford_path):
    word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('resource/GoogleNews-vectors-negative300.bin', binary=True)
    seType_dict = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']

    stanford_nlp = StanfordCoreNLP(stanford_path)

    label_embedding_list = []

    for label in seType_dict:
        # print("label: ", label)
        word_embeddings_list = from_sentence_2_word_embeddings_list(label, stanford_nlp, word2vec_vocab).cuda()
        # print("word_embeddings_list.shape: ", word_embeddings_list.shape)
        label_embedding = torch.max(word_embeddings_list, dim=0)[0]   # will get size is 300
        # print("label_embedding.shape: ", label_embedding.shape)
        label_embedding_list.append(label_embedding)
        # input()

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
        self.BiLSTM_2 = nn.LSTM(300,
                                300 // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)

        self.hidden2tag = nn.Linear(300, 7)
        self.log_softmax = nn.LogSoftmax()

        self.label_embedding_list = do_label_embedding(stanford_path)

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

        # softmax_output = self.softmax(output)  # size is 3 * 7

        pre_labels_list = []
        for i in range(log_softmax_output.shape[0]):
            pre_labels_list.append(int(torch.argmax(log_softmax_output[i])))

        loss = 0
        for j in range(len(gold_labels_list)):
            gold_label = gold_labels_list[j]
            pre_label = pre_labels_list[j]
            loss += -log_softmax_output[j][gold_label]

            sentence_embedding_new = sentence_embeddings_output[j, :]  # size is 300
            loss += -torch.cosine_similarity(sentence_embedding_new,
                                             self.label_embedding_list[gold_label],
                                             dim=0)
            if pre_label != gold_label:
                loss += torch.cosine_similarity(sentence_embedding_new,
                                                   self.label_embedding_list[pre_label],
                                                   dim=0)

        return pre_labels_list, loss
