#!/bin/bash env

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

# p = 0.5
CUDA_VISIBLE_DEVICES=0 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 0 --p 0.5 > output/paragraph_level_BiLSTM/sim_0.5/log/log_paragraph_level_BiLSTM_300_sim_0.5_0.txt &

CUDA_VISIBLE_DEVICES=0 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 1 --p 0.5 > output/paragraph_level_BiLSTM/sim_0.5/log/log_paragraph_level_BiLSTM_300_sim_0.5_1.txt &

CUDA_VISIBLE_DEVICES=0 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 2 --p 0.5 > output/paragraph_level_BiLSTM/sim_0.5/log/log_paragraph_level_BiLSTM_300_sim_0.5_2.txt &

# p = 1
CUDA_VISIBLE_DEVICES=5 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 0 --p 1.0 > output/paragraph_level_BiLSTM/sim_1.0/log/log_paragraph_level_BiLSTM_300_sim_1.0_0.txt &

CUDA_VISIBLE_DEVICES=5 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 1 --p 1.0 > output/paragraph_level_BiLSTM/sim_1.0/log/log_paragraph_level_BiLSTM_300_sim_1.0_1.txt &

CUDA_VISIBLE_DEVICES=5 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 2 --p 1.0 > output/paragraph_level_BiLSTM/sim_1.0/log/log_paragraph_level_BiLSTM_300_sim_1.0_2.txt &


# p = 2
CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 0 --p 2.0 > output/paragraph_level_BiLSTM/sim_2.0/log/log_paragraph_level_BiLSTM_300_sim_2.0_0.txt &

CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 1 --p 2.0 > output/paragraph_level_BiLSTM/sim_2.0/log/log_paragraph_level_BiLSTM_300_sim_2.0_1.txt &

CUDA_VISIBLE_DEVICES=7 nohup \
  python run_paragraph_level_BiLSTM_try_sim.py --fold_num 2 --p 2.0 > output/paragraph_level_BiLSTM/sim_2.0/log/log_paragraph_level_BiLSTM_300_sim_2.0_2.txt &