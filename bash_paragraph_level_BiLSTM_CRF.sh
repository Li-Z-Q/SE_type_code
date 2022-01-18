#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_CRF.py --fold_num 0 > output/paragraph_level_BiLSTM/CRF/log/log_paragraph_level_BiLSTM_300_0.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_CRF.py --fold_num 1 > output/paragraph_level_BiLSTM/CRF/log/log_paragraph_level_BiLSTM_300_1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_CRF.py --fold_num 2 > output/paragraph_level_BiLSTM/CRF/log/log_paragraph_level_BiLSTM_300_2.txt &

