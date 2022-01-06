import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForTokenClassification


def do_config():
    model_config = BertConfig.from_pretrained('pre_train')
    model_config.num_labels = 7
    # model_config.output_attentions = True
    model_config.output_hidden_states = True
    return model_config


print("paragraph level BERT try sim")


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.model_config = do_config()

        self.bert_model_1 = BertModel.from_pretrained('pre_train/', config=self.model_config)
        self.bert_model_2 = BertForTokenClassification.from_pretrained('pre_train/', config=self.model_config)

        self.sim_softmax = nn.Softmax(dim=0)
        self.reset_num = 0
        self.correct_representation_list = torch.tensor([[0.0 for _ in range(768)] for __ in range(7)]).cuda()  # each class a correct representation
        self.correct_num_list = [1] * 7
        self.last_epoch_correct_representation_list = torch.tensor([[0.0 for __ in range(768)] for _ in range(7)]).cuda()  # each class a correct representation
        self.sim_matrix = [[0 for _ in range(7)] for __ in range(7)]

    def forward(self, sentences_list, gold_labels_list):

        # print("before bert1: ", torch.cuda.memory_allocated(0))

        if len(sentences_list) > 512:
            gold_labels_list = gold_labels_list[:512]
            sentences_list = sentences_list[:512]

        sentence_embeddings_list = []
        for word_ids_list in sentences_list:
            word_embeddings_output_list = self.bert_model_1(
                word_ids_list.cuda()).last_hidden_state  # 1 * sentence_len * 768
            sentence_embedding = word_embeddings_output_list[0, 0, :]  # [CLS]'s output embedding,
            sentence_embeddings_list.append(sentence_embedding)

        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 768
        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 768

        gold_labels_list = torch.tensor(gold_labels_list).cuda()
        outputs = self.bert_model_2(inputs_embeds=sentence_embeddings_list, labels=gold_labels_list)

        logits = outputs.logits.squeeze(0)  # sentence_num * 7
        pre_labels_list = []
        for i in range(logits.shape[0]):
            pre_labels_list.append(int(torch.argmax(logits[i])))

        loss = outputs.loss
        last_hidden_states = outputs.hidden_states[-1]  # 1 * s.num * 768
        last_hidden_states = last_hidden_states.squeeze(0)

        for j in range(len(gold_labels_list)):
            pre_label = pre_labels_list[j]
            gold_label = gold_labels_list[j]
            sentence_embedding_new = last_hidden_states[j, :]  # size is 768
            if self.reset_num > 1:
                # ###################################################### check if label is related with sim
                sim_list = []
                for i in range(7):
                    sim_list.append(torch.cosine_similarity(sentence_embedding_new,
                                                            self.last_epoch_correct_representation_list[i, :],
                                                            dim=0))
                # sim_list = self.sim_softmax(torch.tensor(sim_list))
                for i in range(7):
                    self.sim_matrix[gold_label][i] += sim_list[i]
                # ###########################################################################################################

                if pre_label != gold_label:
                    sim_loss = torch.cosine_similarity(sentence_embedding_new,
                                                       self.last_epoch_correct_representation_list[pre_label, :],
                                                       dim=0)
                    loss += sim_loss

            if pre_label == gold_label:
                self.correct_representation_list[gold_label] = self.correct_representation_list[gold_label] + \
                                                               sentence_embedding_new
                self.correct_num_list[gold_label] += 1

        return pre_labels_list, loss

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
        self.correct_representation_list = torch.tensor([[0.0 for _ in range(768)] for _ in range(7)]).cuda()  # each class a gold representation

        self.correct_representation_list = self.correct_representation_list.detach()
        self.last_epoch_correct_representation_list = self.last_epoch_correct_representation_list.detach()

        self.sim_matrix = [[0 for _ in range(7)] for __ in range(7)]

        # if self.correct_representation_list.requires_grad:
        #     print("retain_grad")
        #     self.correct_representation_list = self.correct_representation_list.retain_grad()
        #     self.last_epoch_correct_representation_list = self.last_epoch_correct_representation_list.retain_grad()