#!/bin/bash

# Run main_Retrieval_Retro.py with 3 different difficulties using nohup for background processes
nohup python main_Retrieval_Retro.py --seed 42 --difficulty easy > run_easy_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 42 --difficulty medium > run_medium_seed42.log 2>&1 &
nohup python main_Retrieval_Retro.py --seed 42 --difficulty hard > run_hard_seed42.log 2>&1 &

