#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

############################################################ cheat = True, mask_p = 0.0
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p 0.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p 0.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p 0.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
#
#
#
############################################################ cheat = True, mask_p = 0.25
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p 0.25 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.25/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p 0.25 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.25/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p 0.25 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.25/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
#
#
#
#
#
############################################################ cheat = True, mask_p = 0.5
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p 0.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.5/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p 0.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.5/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p 0.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.5/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
#
#
#
#
#
############################################################ cheat = True, mask_p = 0.75
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p 0.75 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.75/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p 0.75 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.75/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p 0.75 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_0.75/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
#
#
#
#
#
############################################################ cheat = True, mask_p = -1.0
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p -1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-1.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p -1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-1.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p -1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-1.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
#
#
#
#
############################################################ cheat = True, mask_p = -2.0
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p -2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-2.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p -2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-2.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p -2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-2.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &




########################################################### cheat = True, mask_p = -2.5
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p -2.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-2.5/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p -2.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-2.5/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p -2.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-2.5/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &




############################################################ cheat = True, mask_p = -3.0
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat True --mask_p -3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-3.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat True --mask_p -3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-3.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat True --mask_p -3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_True_-3.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &


#
#

########################################################### cheat = False, mask_p = 0.0
CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
    --fold_num 0 --cheat False --mask_p 0.0 \
  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_0.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &

CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
    --fold_num 1 --cheat False --mask_p 0.0 \
  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_0.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &

CUDA_VISIBLE_DEVICES=3 nohup \
  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
    --fold_num 2 --cheat False --mask_p 0.0 \
  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_0.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
##
##
##
############################################################ cheat = False, mask_p = 1.0
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat False --mask_p 1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_1.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat False --mask_p 1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_1.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat False --mask_p 1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_1.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
#
#
#
############################################################ cheat = False, mask_p = 2.0
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat False --mask_p 2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_2.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat False --mask_p 2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_2.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat False --mask_p 2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_2.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &


############################################################ cheat = False, mask_p = 3.0
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat False --mask_p 3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_3.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat False --mask_p 3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_3.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat False --mask_p 3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_3.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &
#
#
#
############################################################ cheat = False, mask_p = 4.0
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 0 --cheat False --mask_p 4.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_4.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 1 --cheat False --mask_p 4.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_4.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre_large.py \
#    --fold_num 2 --cheat False --mask_p 4.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_large_False_4.0/log/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_large_2.txt &