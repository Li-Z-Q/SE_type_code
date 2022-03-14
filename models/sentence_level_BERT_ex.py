import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification

# tokenizer = BertTokenizer.from_pretrained('pre_train')


print("sentence level BERT extra")


class MyModel(nn.Module):
    def __init__(self, two_C):
        super(MyModel, self).__init__()
        model_config = BertConfig.from_pretrained('pre_train')
        if two_C:
            model_config.num_labels = 2
        else:
            model_config.num_labels = 7
        # model_config.output_attentions = True
        model_config.output_hidden_states = True
        self.bert_model = BertForSequenceClassification.from_pretrained('pre_train/', config=model_config)

        # self.hidden2tag = nn.Linear(768, 7)
        # self.softmax = nn.LogSoftmax()

    def forward(self, inputs, label):  # inputs.shape:  torch.Size([1, 14])
        labels = torch.tensor(label).unsqueeze(0)  # size is 1
        outputs = self.bert_model(inputs, labels=labels.cuda())

        loss = outputs.loss
        output = outputs.logits  # torch.Size([1, 2])
        output = output.squeeze(0)  # size is class_num

        last_hidden_states = outputs.hidden_states[-1]  # torch.Size([1, 14, 768])
        sentence_embedding = last_hidden_states[0, 0, :]  # size is 768
        pre_label = int(torch.argmax(output))

        return pre_label, loss, sentence_embedding, output

    def load_model(self, path):
        return torch.load(path)
