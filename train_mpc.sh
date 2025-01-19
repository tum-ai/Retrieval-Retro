#!/bin/bash

# Aktiviere conda environment falls nötig
source /home/thorben/miniconda3/etc/profile.d/conda.sh  # Pfad anpassen
conda activate retrieval-retro

# Starte Training und logge Output
python train_mpc.py --eval 5 --device 1 --use_our_data > training_log_lr_0005_eval.txt 2>&1
