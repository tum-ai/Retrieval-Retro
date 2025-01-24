#!/bin/bash

# Run main_Retrieval_Retro.py with 3 different difficulties using nohup for background processes
mkdir -p logs
nohup python main_Retrieval_Retro.py --seed 42 --difficulty easy --checkpoint_path checkpoints/easy/epoch150_top5_acc_0.6567_42.pt >> logs/run_easy_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 42 --difficulty medium --checkpoint_path checkpoints/medium/epoch350_top5_acc_0.4305_42.pt >> logs/run_medium_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 42 --difficulty hard --checkpoint_path checkpoints/hard/epoch450_top5_acc_0.4008_42.pt >> logs/run_hard_seed42.log 2>&1 &

nohup python main_Retrieval_Retro.py --seed 69 --difficulty easy > logs/run_easy_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 69 --difficulty medium > logs/run_medium_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 69 --difficulty hard > logs/run_hard_seed42.log 2>&1 &

nohup python main_Retrieval_Retro.py --seed 777 --difficulty easy > logs/run_easy_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 777 --difficulty medium > logs/run_medium_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 777 --difficulty hard > logs/run_hard_seed42.log 2>&1 &