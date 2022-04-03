# -*-coding:utf-8-*-
import os
import json
import torch
from transformers import BertTokenizer
from model.sentence_level_BERT import MyModel

pre_train_path = 'pre_train_cn'


def do_classification(data):
    # seType_dict = ["STATE", "EVENT", "REPORT", "GENERIC_SENTENCE", "GENERALIZING_SENTENCE", "QUESTION", "IMPERATIVE"]
    seType_dict = ["no_event", "event"]

    if type(data) is str:
        data = {'sentence': data}
    if type(data) is dict:
        data = [data]

    model_for_prediction = MyModel(dropout=0.5, num_labels=2, pre_train_path=pre_train_path).load(os.getcwd() + '/server_api/BERT_1_cn.pt')
    tokenizer = BertTokenizer.from_pretrained(pre_train_path)

    for i in range(len(data)):
        sentence = data[i]['sentence']
        inputs = tokenizer(sentence, return_tensors="pt").input_ids.cuda()
        print("inputs.shape: ", inputs.shape)
        label = torch.tensor(0).unsqueeze(0).cuda()

        pre_label, loss = model_for_prediction(inputs, label)
        label = seType_dict[pre_label]
        data[i]['situation entity type'] = label

    return data


if __name__ == '__main__':

    data0 = {'sentence': "马龙赢了"}
    data1 = [
        {'sentence': "马龙赢了"},
        {'sentence': "地球有多大"}]
    data2 = "地球有多大"
    data3 = [{'sentence': "地球有多大"}]


    data = data2
    print("input: ", json.dumps(data, sort_keys=False, indent=4, separators=(',', ':'), ensure_ascii=False))
    # try:
    result = do_classification(data)  # from str to [dict, dict, dict, ...]
    print("type(result): ", type(result))
    print(result)
    print(json.dumps(result, sort_keys=False, indent=4, separators=(',', ':'), ensure_ascii=False))
    # except:
    #     print('try again')