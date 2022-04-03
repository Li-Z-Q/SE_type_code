# situation entity type

## catalogue
```bash
├── README.md
├── server_api
│   ├── annotate_data.py
│   ├── BERT_1_cn.pt
│   ├── model
│   │   └── sentence_level_BERT.py
│   ├── offline_demo.py
│   ├── online_demo.py
│   ├── pre_train_cn
│   ├── run_sentence_level_BERT_cn.py
│   ├── server_api.py
│   ├── tools
│   │   ├── devide_train_batch.py
│   │   ├── get_sentence_level_data.py
│   │   └── print_evaluation_result.py
│   └── train_valid_test
│       ├── test_sentence_level_model.py
│       └── train_valid_sentence_level_model.py
├── data
├── output
├── prepare_data.py
├── requirements.txt
├── models
│   ├── paragraph_level_BiLSTM_label_embedding.py
│   ├── paragraph_level_BiLSTM.py
│   ├── sentence_level_BiLSTM_author.py
│   └── sentence_level_BiLSTM.py
├── tools
│   ├── from_paragraph_to_sentence.py
│   ├── load_data_from_pt.py
│   ├── print_evaluation_result.py
│   └── set_train_batch.py
├── pre_train
├── pre_train_cn
├── resource
│   ├── explicit_connective.txt
│   ├── GoogleNews-vectors-negative300.bin
│   ├── masc_sentence_pos_ner_dict.pkl
│   ├── paragraph_memory.pt
│   ├── statistic_dict_plus.pt
│   ├── statistic_dict_plus_test.pt
│   └── statistic_dict_plus_train.pt
├── run_paragraph_level_BiLSTM_label_embedding.py
├── run_paragraph_level_BiLSTM.py
├── run_sentence_level_BiLSTM_author.py
├── run_sentence_level_BiLSTM.py
└── train_test
    ├── train_valid_paragraph_level_model.py
    └── train_valid_sentence_level_model.py

```

## 1 Chinese server api
### 1.0 data annotation
```bash
conda activate py38
python server_api/annotate_data
```

### 1.1 train
```bash
conda activate py38
python server_api/run_sentence_level_BERT_cn.py
```

### 1.2 start server api
```bash
conda activate py38
python server_api/server_api.py
```

### 1.3 demo
```bash
conda activate py38
python server_api/online_demo.py
python server_api/offline_demo.py
```

## 2 English data, try to get better performence
### 2.0 prepare data
```bash
# do embedding and save as .pt
# use 343 dim: 300 word embedding + 7 ner + 36 pos 
conda activate py27
python prepare_data.py
```
##
the following will base on py38
```bash
conda activate py38
pip install -r requirements.txt
```
### 2.1 sentence level base
```bash
# real sentence level
python run_sentence_level_BiLSTM.py --IF_USE_EX_INITIAL 0
# fake sentence level
python run_sentence_level_BiLSTM_author.py --IF_USE_EX_INITIAL 0
```

### 2.2 paragraph level base
```bash
# base on real sentence level
python run_paragraph_level_BiLSTM.py --IF_USE_EX_INITIAL_1 0 --IF_USE_EX_INITIAL_2 0
# base on fake sentence level
python run_paragraph_level_BiLSTM.py --METHOD 1 --IF_USE_EX_INITIAL_1 0 --IF_USE_EX_INITIAL_2 0
```
##
the following will base on real sentence level
### 2.3 paragraph level with initialized 1st Bi-LSTM
```bash
# freeze 1st Bi-LSTM
python run_paragraph_level_BiLSTM.py --IF_USE_EX_INITIAL_1 1 --FREEZE 1 
# L1 loss
python run_paragraph_level_BiLSTM.py --IF_USE_EX_INITIAL_1 1 --EX_LOSS 1 
# no L1 loss no freeze, tune together with paragraph_level
python run_paragraph_level_BiLSTM.py --IF_USE_EX_INITIAL_1 1
```

### 2.4 paragraph level with initialized 1st Bi-LSTM with label embedding
```bash
# no L1 loss no freeze, tune together with paragraph level
python run_paragraph_level_BiLSTM_label_embedding.py --IF_USE_EX_INITIAL_1 1
```




