import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('pre_train')
model_config = BertConfig.from_pretrained('pre_train')
model_config.num_labels = 7
# model_config.output_attentions = True
# model_config.output_hidden_states = True


print("sentence level BERT")


class MyModel(nn.Module):
    def __init__(self, dropout):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.bert_model = BertForSequenceClassification.from_pretrained('pre_train/', config=model_config)

        # self.hidden2tag = nn.Linear(768, 7)
        # self.softmax = nn.LogSoftmax()

    def forward(self, inputs, label):
        # word_ids_list = word_ids_list.unsqueeze(0)  # 1 * len(word_ids_list)
        #
        # print(word_ids_list.shape)

        labels = torch.tensor(label).unsqueeze(0)  # Batch size 1

        # print(labels)

        outputs = self.bert_model(inputs, labels=labels.cuda())

        loss = outputs.loss
        output = outputs.logits
        output = output.squeeze(0)

        pre_label = int(torch.argmax(output))

        return pre_label, loss
