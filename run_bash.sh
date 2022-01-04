#!/bin/bash env
# get all filename in specified path

export PYTHONUNBUFFERED=1
# export PYTHONPATH=${PYTHONPATH}:./


# python run_sentence_level_BERT.py
# python run_sentence_level_BiLSTM.py


# python run_paragraph_level_BERT.py
python run_paragraph_level_BiLSTM.py
 
# python run_paragraph_level_BERT_CRF.py
# nohup python run_paragraph_level_BiLSTM_CRF.py > output/log_paragraph_level_BiLSTM_CRF.txt & 