#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

CUDA_VISIBLE_DEVICES=0 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 0 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_0.txt &
CUDA_VISIBLE_DEVICES=0 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 1 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_1.txt &
CUDA_VISIBLE_DEVICES=0 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 2 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_2.txt &
CUDA_VISIBLE_DEVICES=5 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 3 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_3.txt &
CUDA_VISIBLE_DEVICES=5 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 4 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_4.txt &
CUDA_VISIBLE_DEVICES=5 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 5 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_5.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 6 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_6.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 7 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_7.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 8 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_8.txt &
CUDA_VISIBLE_DEVICES=7 nohup python run_paragraph_level_BiLSTM_try_sim.py --fold_num 9 > output/paragraph_level_BiLSTM/sim/log/log_paragraph_level_BiLSTM_300_sim_9.txt &
