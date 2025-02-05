#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate retrieval-retro
# Run main_Retrieval_Retro.py with 3 different difficulties using nohup for background processes
mkdir -p logs
nohup python main_Retrieval_Retro.py --seed 42 --difficulty easy > logs/run_easy_seed42_naive.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 42 --difficulty medium > logs/run_medium_seed42_naive.log 2>&1 &
# nohup python main_Retrieval_Retro.py --seed 42 --difficulty hard --checkpoint_path checkpoints/RR/hard/epoch450_top5_acc_0.4008_42.pt > logs/run_hard_seed42.log 2>&1 &

# nohup python main_Retrieval_Retro.py --seed 69 --difficulty easy > logs/run_easy_seed69_naive.log 2>&1 &
# nohup python main_Retrieval_Retro.py --seed 777 --difficulty easy > logs/run_easy_seed777_naive.log 2>&1 &
# nohup python main_Retrieval_Retro.py --seed 888 --difficulty easy > logs/run_easy_seed888_naive.log 2>&1 &

# nohup python main_Retrieval_Retro.py --seed 69 --difficulty medium > logs/run_medium_seed69_naive.log 2>&1 &
# nohup python main_Retrieval_Retro.py --seed 777 --difficulty medium > logs/run_medium_seed777_naive.log 2>&1 &
# nohup python main_Retrieval_Retro.py --seed 888 --difficulty medium > logs/run_medium_seed888_naive.log 2>&1 &

# nohup python main_Retrieval_Retro.py --seed 69 --difficulty hard > logs/run_hard_seed69_naive.log 2>&1 &
# nohup python main_Retrieval_Retro.py --seed 777 --difficulty hard > logs/run_hard_seed777_naive.log 2>&1 &
# nohup python main_Retrieval_Retro.py --seed 888 --difficulty hard > logs/run_hard_seed888_naive.log 2>&1 &

nohup python main_Retrieval_Retro.py --seed 999 --difficulty easy > logs/run_easy_seed999_naive.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 999 --difficulty medium > logs/run_medium_seed999_naive.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 999 --difficulty hard > logs/run_hard_seed999_naive.log 2>&1 &
