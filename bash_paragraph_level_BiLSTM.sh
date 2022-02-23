#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

CUDA_VISIBLE_DEVICES=7 nohup python run_paragraph_level_BiLSTM.py --fold_num 0 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_0.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_paragraph_level_BiLSTM.py --fold_num 1 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_1.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_paragraph_level_BiLSTM.py --fold_num 2 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_2.txt &
#CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM.py --fold_num 3 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_3.txt &
#CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM.py --fold_num 4 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_4.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM.py --fold_num 5 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_5.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM.py --fold_num 6 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_6.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM.py --fold_num 7 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_7.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM.py --fold_num 8 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_8.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM.py --fold_num 9 > output/paragraph_level_BiLSTM/base/log/log_paragraph_level_BiLSTM_300_9.txt &
