#!/bin/bash env

source activate py38
export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

## sentence_base
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_sentence_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL 0 \
#  > output/sentence_base.txt &

# paragraph_author
CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM.py \
    --METHOD 1 \
    --IF_USE_EX_INITIAL_1 0 \
    --IF_USE_EX_INITIAL_2 0 \
  > output/paragraph_author.txt &

# paragraph_base
CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM.py \
    --IF_USE_EX_INITIAL_1 0 \
    --IF_USE_EX_INITIAL_2 0 \
  > output/paragraph_base.txt &

#
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
#
## paragraph_double_initial_loss
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM.py \
#    --IF_USE_EX_INITIAL_1 0 \
#    --IF_USE_EX_INITIAL_2 1 \
#    --EX_LOSS 1 \
#  > output/paragraph_double_initial_loss.txt &
