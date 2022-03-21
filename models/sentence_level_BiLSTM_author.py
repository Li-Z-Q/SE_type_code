import torch
import torch.nn as nn
from torch.autograd import Variable


class MyModel(nn.Module):
    def __init__(self, input_dim, dropout, random_seed, if_use_ex_initial):
        super(MyModel, self).__init__()
        print("sentence level BiLSTM author")

        self.random_seed = random_seed
        self.if_use_ex_initial = if_use_ex_initial

        self.dropout = nn.Dropout(p=dropout)

        self.BiLSTM = nn.LSTM(input_dim, 300 // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.hidden2tag = nn.Linear(300, 7)
        self.softmax = nn.LogSoftmax()

    def forward(self, sentences_list):  # [4*3336, 7*336, 1*336]
        len_list = [sentence.shape[0] for sentence in sentences_list]

        sentences_list = torch.cat([sentence for sentence in sentences_list], dim=0)  # (4+7+1) * 348
        sentences_list = sentences_list.unsqueeze(0)
        sentences_list = self.dropout(sentences_list)

        all_word_embeddings_output, _ = self.BiLSTM(sentences_list, (Variable(torch.zeros(2, 1, 150)).cuda(), Variable(torch.zeros(2, 1, 150)).cuda()))  # 1 * sentence_len * 300

        start_position = 0
        sentence_embeddings_list = []
        for i in range(len(len_list)):
            sentence_embedding = all_word_embeddings_output[0, start_position:start_position+len_list[i], :]  # sentence_len * 300
            sentence_embedding = torch.max(sentence_embedding, dim=0)[0]  # size is 300
            sentence_embeddings_list.append(sentence_embedding)
            start_position = start_position+len_list[i]

        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 300
        output = self.softmax(self.hidden2tag(sentence_embeddings_list))  # sentence_num * 7
        pre_labels_list = [int(torch.argmax(output[i])) for i in range(output.shape[0])]

        return [pre_labels_list, []], [output, []], sentence_embeddings_list

    def save(self):
        torch.save(self, 'models/model_sentence_level_BiLSTM_author_' + str(self.random_seed) + '.pt')

    def load(self):
        return torch.load('models/model_sentence_level_BiLSTM_author_' + str(self.random_seed) + '.pt')
