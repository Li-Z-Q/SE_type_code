#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 0 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_0.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 1 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 2 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_2.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 3 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_3.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 4 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_4.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 5 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_5.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 6 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_6.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 7 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_7.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 8 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_8.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_paragraph_level_BiLSTM_label_embedding_MLP_top.py --fold_num 9 > output/paragraph_level_BiLSTM/label_embedding_MLP_top/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_top_9.txt &