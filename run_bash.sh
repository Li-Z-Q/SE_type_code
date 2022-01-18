#!/bin/bash env
#get all filename in specified path

export PYTHONUNBUFFERED=1
#export PYTHONPATH=${PYTHONPATH}:./

#export DET_MASTER="124.16.138.147:8080"
#det user login lizhuoqun
#nohup det cmd run --config-file config.yaml -c ./ bash run_bash.sh > output/paragraph_level_BERT/base/log/log_paragraph_level_BERT_1 2>&1 &


#python test_out.py
#python run_sentence_level_BERT.py
#python run_sentence_level_BiLSTM.py


python run_paragraph_level_BERT.py --fold_num 1
#python run_paragraph_level_BiLSTM.py
 
#python run_paragraph_level_BERT_CRF.py
#python run_paragraph_level_BiLSTM_CRF.py

#python run_paragraph_level_BERT_try_sim.py
#python run_paragraph_level_BERT_try_sim_middle.py