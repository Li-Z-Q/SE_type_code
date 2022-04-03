#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

source activate py38

CUDA_VISIBLE_DEVICES=7 nohup python run_sentence_level_BERT_cn.py > train_log.txt &