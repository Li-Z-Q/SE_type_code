import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel, BertForTokenClassification

tokenizer = BertTokenizer.from_pretrained('pre_train')
model_config = BertConfig.from_pretrained('pre_train')
model_config.num_labels = 7
# model_config.output_attentions = True
# model_config.output_hidden_states = True


print("paragraph level BERT")


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.bert_model_1 = BertModel.from_pretrained('pre_train/', config=model_config)
        self.bert_model_2 = BertForTokenClassification.from_pretrained('pre_train/', config=model_config)

    def forward(self, sentences_list, gold_labels_list):

        # print("before bert1: ", torch.cuda.memory_allocated(0))

        if len(sentences_list) > 512:
            gold_labels_list = gold_labels_list[:512]
            sentences_list = sentences_list[:512]

        sentence_embeddings_list = []
        for sentence in sentences_list:
            word_ids_list = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # 1 * sentence_len
            word_embeddings_output_list = self.bert_model_1(word_ids_list.cuda()).last_hidden_state  # 1 * sentence_len * 768
            sentence_embedding = word_embeddings_output_list[0, 0, :]  # [CLS]'s output embedding,
            sentence_embeddings_list.append(sentence_embedding)
            # print("after a sentence: ", torch.cuda.memory_allocated(0))
            # if torch.cuda.memory_allocated(0) > 10000000000:
            #     gold_labels_list = gold_labels_list[:len(sentence_embeddings_list)]
            #     break

        sentence_embeddings_list = torch.stack(sentence_embeddings_list)  # sentence_num * 768
        # sentence_embeddings_list = torch.cat((CLS_SEP[:, 0], sentence_embeddings_list))
        # sentence_embeddings_list = torch.cat((sentence_embeddings_list, CLS_SEP[:, 1]))
        sentence_embeddings_list = sentence_embeddings_list.unsqueeze(0)  # 1 * sentence_num * 768

        # print("before bert2: ", torch.cuda.memory_allocated(0))
        gold_labels_list = torch.tensor(gold_labels_list).cuda()
        outputs = self.bert_model_2(inputs_embeds=sentence_embeddings_list, labels=gold_labels_list)

        # print("after bert2: ", torch.cuda.memory_allocated(0))

        loss = outputs.loss
        logits = outputs.logits.squeeze(0)  # sentence_num * 7
        
        pre_labels_list = []
        for i in range(logits.shape[0]):
            pre_labels_list.append(int(torch.argmax(logits[i])))

        return pre_labels_list, loss