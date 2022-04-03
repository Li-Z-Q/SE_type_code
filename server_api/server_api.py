import os
import sys
sys.path.append(os.getcwd() + '/model')
sys.path.append(os.getcwd() + '/pre_train_cn')
# print(sys.path)

import json
import torch
import uvicorn
from model.sentence_level_BERT import MyModel

from fastapi import FastAPI
from transformers import BertTokenizer

app = FastAPI()
# seType_dict = ["STATE", "EVENT", "REPORT", "GENERIC_SENTENCE", "GENERALIZING_SENTENCE", "QUESTION", "IMPERATIVE"]
seType_dict = ["no_event", "event"]

pre_train_path = 'pre_train_cn'
# export PYTHONUNBUFFERED = 1

@app.get("/user/{data}")
def classification_fn(data:str):

    data = json.loads(data)  # data is [dict, dict, ...] or a dict or a sentence
    if type(data) is str:
        data = {'sentence': data}
    if type(data) is dict:
        data = [data]

    for i in range(len(data)):
        sentence = data[i]['sentence']
        inputs = tokenizer(sentence, return_tensors="pt").input_ids.cuda()
        label = torch.tensor(0).unsqueeze(0).cuda()

        pre_label, loss = model_for_prediction.forward(inputs, label)

        label = seType_dict[pre_label]
        data[i]['situation entity type'] = label

    return data


model_for_prediction = MyModel(dropout=0.5, num_labels=2, pre_train_path=pre_train_path).load(os.getcwd() + '/server_api/BERT_1_cn.pt')
tokenizer = BertTokenizer.from_pretrained(pre_train_path)
if __name__ == '__main__':
    uvicorn.run(app='server_api:app', host="0.0.0.0", port=8000, reload=True)
    # uvicorn.run(app='server_api:app', host="192.168.14.14", port=8855, reload=True)