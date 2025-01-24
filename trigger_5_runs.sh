#!/bin/bash

# Run main_Retrieval_Retro.py with 3 different difficulties using nohup for background processes
mkdir -p logs
nohup python main_Retrieval_Retro.py --seed 42 --difficulty easy --checkpoint_path checkpoints/RR/easy/epoch100_top5_acc_0.6762_42.pt >> logs/run_easy_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 42 --difficulty medium --checkpoint_path checkpoints/RR/medium/epoch250_top5_acc_0.4391_42.pt >> logs/run_medium_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 42 --difficulty hard --checkpoint_path checkpoints/RR/hard/epoch450_top5_acc_0.4008_42.pt >> logs/run_hard_seed42.log 2>&1 &

nohup python main_Retrieval_Retro.py --seed 69 --difficulty easy > logs/run_easy_seed69.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 69 --difficulty medium > logs/run_medium_seed69.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 69 --difficulty hard > logs/run_hard_seed69.log 2>&1 &

nohup python main_Retrieval_Retro.py --seed 777 --difficulty easy > logs/run_easy_seed777.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 777 --difficulty medium > logs/run_medium_seed777.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 777 --difficulty hard > logs/run_hard_seed777.log 2>&1 &