#!/bin/bash env

source activate py38
export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

## sentence_base
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_sentence_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_base.txt &

## sentence_author
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_sentence_level_BiLSTM_author.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_author.txt &

## paragraph_base
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 0 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_base.txt &

## paragraph_author
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --METHOD 1 \
#    --IF_USE_EX_INITIAL_1 0 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_author.txt &



## sentence_base_3_class
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_sentence_level_BiLSTM_3_class.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_base_3_class.txt &

## sentence_author_3_class
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_sentence_level_BiLSTM_author_3_class.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_author_3_class.txt &

## sentence_base_Linear_3_class
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_sentence_level_Linear_3_class.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_base_Linear_3_class.txt &



## paragraph_1st_initial_freeze
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#    --FREEZE 1 \
#  > output/paragraph_1st_initial_freeze.txt &

## paragraph_1st_initial_loss
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#    --EX_LOSS 1 \
#  > output/paragraph_1st_initial_loss.txt &
#
## paragraph_1st_initial_non-loss
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_1st_initial_non-loss.txt &

## paragraph_1st_initial_non-loss_label_embedding
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_1st_initial_non-loss_label_embedding.txt &



# paragraph_1st_author_initial
CUDA_VISIBLE_DEVICES=2 nohup \
  python run_paragraph_level_BiLSTM_author_initial.py \
  > output/paragraph_1st_author_initial_non-loss.txt &



## paragraph_1st_initial_non-loss_label_embedding_3_class
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_3_class.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_1st_initial_non-loss_label_embedding_3_class_without_label_all.txt &
##  > output/paragraph_1st_initial_non-loss_label_embedding_3_class.txt &
##  > output/paragraph_1st_initial_non-loss_label_embedding_3_class_without_label_all.txt &



## paragraph_1st_initial_non-loss_label_embedding_3_class_with_double_LSTM
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_3_class_with_double_LSTM.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_1st_initial_non-loss_label_embedding_3_class_with_double_LSTM.txt &




## paragraph_double_initial_loss
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 0 \
#    --IF_USE_EX_INITIAL_2 1 \
#    --EX_LOSS 1 \
#  > output/paragraph_double_initial_loss.txt &