#!/bin/bash env
source activate py38
export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./
##
############################################################ cheat = True, mask_p = 0.0
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p 0.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p 0.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p 0.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
#
#
############################################################ cheat = True, mask_p = 0.25
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p 0.25 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.25/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p 0.25 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.25/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p 0.25 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.25/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
####
####
####
####
####
############################################################ cheat = True, mask_p = 0.5
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p 0.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.5/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p 0.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.5/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p 0.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.5/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
##
###

#
############################################################ cheat = True, mask_p = 0.5, control = T
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p 0.5 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.5_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p 0.5 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.5_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p 0.5 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.5_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
##
#
####
###
############################################################ cheat = True, mask_p = 0.75
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p 0.75 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.75/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p 0.75 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.75/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p 0.75 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.75/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
##
##
##
##
##
############################################################ cheat = True, mask_p = -1.0
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p -1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-1.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p -1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-1.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p -1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-1.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
##

############################################################ cheat = True, mask_p = -1.0 grad = F
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p -1.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-1.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p -1.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-1.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=2 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p -1.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-1.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
####


##
##
############################################################ cheat = True, mask_p = -2.0
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p -2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-2.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p -2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-2.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p -2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-2.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
#
#
#
########################################################### cheat = True, mask_p = -2.5
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p -2.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-2.5/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p -2.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-2.5/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p -2.5 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-2.5/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
#
#
###
############################################################ cheat = True, mask_p = -100, control = T
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p -100 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-100_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p -100 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-100_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p -100 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-100_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
#
############################################################# cheat = True, mask_p = -104, , control = T
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --EPOCHs 10 --fold_num 0 --cheat True --mask_p -104 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-104_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --EPOCHs 10 --fold_num 1 --cheat True --mask_p -104 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-104_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --EPOCHs 10 --fold_num 2 --cheat True --mask_p -104 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_-104_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#


############################################################ cheat = True, mask_p = 0.33
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat True --mask_p 0.33 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.33/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat True --mask_p 0.33 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.33/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat True --mask_p 0.33 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_True_0.33/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &

#
#
###
############################################################ cheat = False, mask_p = 0.0
CUDA_VISIBLE_DEVICES=6 nohup \
  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
    --fold_num 0 --cheat False --mask_p 0.0 \
  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &

CUDA_VISIBLE_DEVICES=6 nohup \
  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
    --fold_num 1 --cheat False --mask_p 0.0 \
  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &

CUDA_VISIBLE_DEVICES=6 nohup \
  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
    --fold_num 2 --cheat False --mask_p 0.0 \
  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
###


####
############################################################# cheat = False, mask_p = 0.0, memory = True
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 0.0 --if_use_memory 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_memory/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 0.0 --if_use_memory 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_memory/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 0.0 --if_use_memory 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_memory/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
#
##
#
############################################################# cheat = False, mask_p = 0.0, grad = False
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 0.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 0.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 0.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#


############################################################# cheat = False, mask_p = 0.0, if_control_loss = True
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 0.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 0.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 0.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &

#
############################################################# cheat = False, mask_p = 0.1, if_control_loss = True
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 0.1 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.1_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 0.1 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.1_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 0.1 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_0.1_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#


############################################################# cheat = False, mask_p = 1.0
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_1.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_1.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=7 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 1.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_1.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
###
###
###
############################################################# cheat = False, mask_p = 2.0
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=6 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 2.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
###
##
#
############################################################# cheat = False, mask_p = 2.0, grad = False
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 2.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 2.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 2.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
#
#
############################################################# cheat = False, mask_p = 2.0, if_control_loss = True
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 2.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 2.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 2.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_2.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &

#
############################################################ cheat = False, mask_p = 3.0
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_3.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_3.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 3.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_3.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &

#
##
############################################################ cheat = False, mask_p = 4.0
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 4.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 4.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 4.0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &


############################################################ cheat = False, mask_p = 4.0 grad = False
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 4.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 4.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 4.0 --bilstm_1_grad 0 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0_grad_F/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
#
#
############################################################ cheat = False, mask_p = 4.0 if_control_loss = True
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 0 --cheat False --mask_p 4.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_0.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 1 --cheat False --mask_p 4.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_1.txt &
#
#CUDA_VISIBLE_DEVICES=3 nohup \
#  python run_paragraph_level_BiLSTM_label_embedding_MLP_pre.py \
#    --fold_num 2 --cheat False --mask_p 4.0 --if_control_loss 1 \
#  > output/paragraph_level_BiLSTM/label_embedding_MLP_pre_False_4.0_control_T/log_paragraph_level_BiLSTM_300_label_embedding_MLP_pre_2.txt &
#
