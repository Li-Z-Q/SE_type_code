import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

# model_config.output_attentions = True
# model_config.output_hidden_states = True


print("sentence level BERT")


class MyModel(nn.Module):
    def __init__(self, dropout, num_labels=7, pre_train_path='pre_train'):
        super(MyModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        model_config = BertConfig.from_pretrained(pre_train_path)
        model_config.num_labels = num_labels
        self.bert_model = BertForSequenceClassification.from_pretrained(pre_train_path + '/', config=model_config)

    def forward(self, inputs, label):
        labels = torch.tensor(label).unsqueeze(0)  # Batch size 1, labels is 1 * 1

        outputs = self.bert_model(inputs, labels=labels.cuda())

        loss = outputs.loss
        output = outputs.logits
        output = output.squeeze(0)

        pre_label = int(torch.argmax(output))

        return pre_label, loss

    def save(self, path):
        torch.save(self, path)

    def load(self, path):
        model_for_prediction = torch.load(path)
        return model_for_prediction