import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertTokenizer, BertConfig, BertModel

tokenizer = BertTokenizer.from_pretrained('pre_train')
model_config = BertConfig.from_pretrained('pre_train')
model_config.num_labels = 7
# model_config.output_attentions = True
# model_config.output_hidden_states = True


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.bert_model_1 = BertModel.from_pretrained('pre_train/', config=model_config)
        self.bert_model_2 = BertModel.from_pretrained('pre_train/', config=model_config)

        self.hiden2tag = nn.Linear(768, 7)

        self.crf = CRF(num_tags=7, batch_first=True)

    def forward(self, sentences_list, label_list):

        # print("before bert1: ", torch.cuda.memory_allocated(0))

        if len(sentences_list) > 512:
            label_list = label_list[:512]
            sentences_list = sentences_list[:512]

        sentence_embeddings_list = []
        for sentence in sentences_list:
            word_ids_list = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # 1 * sentence_len
            word_embeddings_output_list = self.bert_model_1(word_ids_list.cuda()).last_hidden_state  # 1 * sentence_len * 768
            sentence_embedding = word_embeddings_output_list[0, 0, :]  # [CLS]'s output embedding,
            sentence_embeddings_list.append(sentence_embedding)
            # print("after a sentence: ", torch.cuda.memory_allocated(0))

        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 768
        # sentence_embeddings_list = torch.cat((CLS_SEP[:, 0], sentence_embeddings_list))
        # sentence_embeddings_list = torch.cat((sentence_embeddings_list, CLS_SEP[:, 1]))
        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 768

        # print("before bert2: ", torch.cuda.memory_allocated(0))
        label_list = torch.tensor(label_list).cuda()
        label_list = label_list.unsqueeze(0)  # 1 * sentence_num

        pro_matrix = self.bert_model_2(inputs_embeds=sentence_embeddings_list).last_hidden_state  # 1 * sentence_num * 768
        pro_matrix = self.dropout(pro_matrix)

        pro_matrix = self.hiden2tag(pro_matrix.squeeze(0)).unsqueeze(0)  # 1 * sentence_num * 7

        loss = -self.crf(pro_matrix, label_list)
        outputs = self.crf.decode(pro_matrix)[0]  # is a list, len == sentence_num

        return outputs, loss