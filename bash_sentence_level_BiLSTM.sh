#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 0 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_0.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 1 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 2 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_2.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 3 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_3.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 4 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_4.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 5 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_5.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 6 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_6.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 7 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_7.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 8 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_8.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py --fold_num 9 > output/sentence_level_BiLSTM/base/log/log_sentence_level_BiLSTM_300_9.txt &
