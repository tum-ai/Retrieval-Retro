#!/bin/bash

# Aktiviere conda environment falls nötig
source /home/thorben/miniconda3/etc/profile.d/conda.sh  # Pfad anpassen
conda activate retrieval-retro

# Starte Training und logge Output
python train_mpc.py --device 0 > ./experiments/training_log_their_results.txt 2>&1
