#!/bin/bash env
#get all filename in specified path

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./


CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM.py > output/sentence_level_bilstm/log_sentence_level_BiLSTM_300.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM_try_sim.py > output/sentence_level_bilstm/log_sentence_level_BiLSTM_300_try_sim.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_sentence_level_BiLSTM_try_sim_weight_average_gold.py > output/sentence_level_bilstm/log_sentence_level_BiLSTM_300_try_sim_weight_average_gold.txt &

CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM.py > output/paragraph_level_bilstm/log_paragraph_level_BiLSTM_300.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_CRF.py > output/paragraph_level_bilstm/log_paragraph_level_BiLSTM_300_CRF.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_try_sim.py > output/paragraph_level_bilstm/log_paragraph_level_BiLSTM_300_try_sim.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_paragraph_level_BiLSTM_try_sim_middle.py > output/paragraph_level_bilstm/log_paragraph_level_BiLSTM_300_try_sim_middle.txt &

CUDA_VISIBLE_DEVICES=5 nohup python run_sentence_level_BERT.py > output/sentence_level_bert/log_sentence_level_BERT.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_sentence_level_BERT_try_sim.py > output/sentence_level_bert/log_sentence_level_BERT_try_sim.txt &
