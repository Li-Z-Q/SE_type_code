#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 0 > output/sentence_level_BiLSTM/base/log_sentence_level_BiLSTM_300_0.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 1 > output/sentence_level_BiLSTM/base/log_sentence_level_BiLSTM_300_1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 2 > output/sentence_level_BiLSTM/base/log_sentence_level_BiLSTM_300_2.txt &
