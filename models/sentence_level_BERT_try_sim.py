import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification


def do_config():
    model_config = BertConfig.from_pretrained('pre_train')
    model_config.num_labels = 7
    # model_config.output_attentions = True
    model_config.output_hidden_states = True

    return model_config


print("sentence level BERT try sim")


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.model_config = do_config()

        self.bert_model = BertForSequenceClassification.from_pretrained('pre_train/', config=self.model_config)

        self.softmax = nn.Softmax(dim=0)

        # self.hidden2tag = nn.Linear(768, 7)
        # self.softmax = nn.LogSoftmax()
        self.reset_num = 0
        self.sim_softmax = nn.Softmax(dim=0)
        self.correct_representation_list = torch.tensor([[0.0 for _ in range(768)] for __ in range(7)]).cuda()  # each class a correct representation
        self.correct_num_list = [1] * 7
        self.last_epoch_correct_representation_list = torch.tensor([[0.0 for __ in range(768)] for _ in range(7)]).cuda()  # each class a correct representation
        self.sim_matrix = [[0 for _ in range(7)] for __ in range(7)]

    def forward(self, inputs, gold_label):
        # word_ids_list = word_ids_list.unsqueeze(0)  # 1 * len(word_ids_list)
        #
        # print(word_ids_list.shape)

        labels = torch.tensor(gold_label).unsqueeze(0)  # Batch size 1

        # print(labels)

        outputs = self.bert_model(inputs, labels=labels.cuda())

        output = outputs.logits
        output = output.squeeze(0)  # size is 7
        pre_label = int(torch.argmax(output))
        softmax_output = self.softmax(output)

        loss = outputs.loss

        last_hidden_states = outputs.hidden_states[-1]  # 1 * s.len * 768
        last_hidden_states_CLS = last_hidden_states.squeeze(0)[0, :]  # size = 768

        if self.reset_num > 1:
            # ###################################################### check if label is related with sim
            sim_list = []
            for i in range(7):
                sim_list.append(torch.cosine_similarity(last_hidden_states_CLS,
                                                        self.last_epoch_correct_representation_list[i, :],
                                                        dim=0))
            # sim_list = self.sim_softmax(torch.tensor(sim_list))
            for i in range(7):
                self.sim_matrix[gold_label][i] += sim_list[i]
            # ###########################################################################################################

            if pre_label != gold_label:
                sim_loss = torch.cosine_similarity(last_hidden_states_CLS,
                                                   self.last_epoch_correct_representation_list[pre_label, :],
                                                   dim=0)
                loss += sim_loss

        if pre_label == gold_label:
            self.correct_representation_list[gold_label] = self.correct_representation_list[gold_label] + \
                                                           last_hidden_states_CLS * softmax_output[gold_label]
            self.correct_num_list[gold_label] += softmax_output[gold_label]

        return pre_label, loss

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
        self.correct_representation_list = torch.tensor(
            [[0.0 for _ in range(768)] for _ in range(7)]).cuda()  # each class a gold representation

        self.correct_representation_list = self.correct_representation_list.detach()
        self.last_epoch_correct_representation_list = self.last_epoch_correct_representation_list.detach()

        self.sim_matrix = [[0 for _ in range(7)] for __ in range(7)]

        # if self.correct_representation_list.requires_grad:
        #     print("retain_grad")
        #     self.correct_representation_list = self.correct_representation_list.retain_grad()
        #     self.last_epoch_correct_representation_list = self.last_epoch_correct_representation_list.retain_grad()
