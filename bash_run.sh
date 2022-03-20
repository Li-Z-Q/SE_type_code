#!/bin/bash env

source activate py38
export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

## paragraph_author
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --METHOD 1 \
#    --IF_USE_EX_INITIAL_1 0 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_author.txt &





## sentence_base
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_sentence_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_base.txt &

## sentence_base_3_class
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_sentence_level_BiLSTM_3_class.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_base_3_class.txt &

## paragraph_base
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 0 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_base.txt &





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

## paragraph_double_initial_loss
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 0 \
#    --IF_USE_EX_INITIAL_2 1 \
#    --EX_LOSS 1 \
#  > output/paragraph_double_initial_loss.txt &




## paragraph_1st_initial_freeze_label_embedding
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#    --FREEZE 1 \
#  > output/paragraph_1st_initial_freeze_label_embedding.txt &
#
## paragraph_1st_initial_loss_label_embedding
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#    --EX_LOSS 1 \
#  > output/paragraph_1st_initial_loss_label_embedding.txt &
#
## paragraph_1st_initial_non-loss_label_embedding
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#  > output/paragraph_1st_initial_non-loss_label_embedding.txt &





## paragraph_1st_initial_freeze_label_embedding_3_class
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_3_class.py \
#    --IF_USE_EX_INITIAL_1 1 \
#    --IF_USE_EX_INITIAL_2 0 \
#    --FREEZE 1 \
#  > output/paragraph_1st_initial_freeze_label_embedding_3_class.txt &

# paragraph_1st_initial_non-loss_label_embedding_3_class
CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM_label_embedding_3_class.py \
    --IF_USE_EX_INITIAL_1 1 \
    --IF_USE_EX_INITIAL_2 0 \
  > output/paragraph_1st_initial_non-loss_label_embedding_3_class.txt &
#  > output/paragraph_1st_initial_non-loss_label_embedding_3_class.txt &
#  > output/paragraph_1st_initial_non-loss_label_embedding_3_class_without_label_1.txt &
#  > output/paragraph_1st_initial_non-loss_label_embedding_3_class_without_label_all.txt &