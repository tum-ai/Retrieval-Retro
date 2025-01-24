#!/bin/bash

# Run main_Retrieval_Retro.py with 5 different seeds and redirect output to log files
python main_Retrieval_Retro.py --seed 42 > run_seed42.log 2>&1
python main_Retrieval_Retro.py --seed 123 > run_seed123.log 2>&1
python main_Retrieval_Retro.py --seed 456 > run_seed456.log 2>&1
python main_Retrieval_Retro.py --seed 789 > run_seed789.log 2>&1
python main_Retrieval_Retro.py --seed 999 > run_seed999.log 2>&1

