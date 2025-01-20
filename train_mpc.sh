#!/bin/bash

# Aktiviere conda environment falls nötig
source /home/thorben/miniconda3/etc/profile.d/conda.sh  # Pfad anpassen
conda activate retrieval-retro

# Starte alle Trainings parallel
echo "Starting training for all difficulties in parallel..."
#python train_mpc.py --eval 5 --device 1 --our_data_difficulty easy > ./experiments/MPC_easy_our_data_log_lr_0005_eval.txt 2>&1 &
#python train_mpc.py --eval 5 --device 0 --our_data_difficulty medium > ./experiments/MPC_medium_our_data_log_lr_0005_eval.txt 2>&1 &
#python train_mpc.py --eval 5 --device 2 --our_data_difficulty hard > ./experiments/MPC_hard_our_data_log_lr_0005_eval.txt 2>&1 &
python train_mpc.py --eval 5 --device 0 > ./experiments/MPC_their_data_log_lr_0005_eval.txt 2>&1 &

