#!/bin/bash

# Easy difficulty
# Easy difficulty
for seed in 69 777 888; do
    python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_naive_updated.json \
                     --output /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_naive_eval.json \
                     --log /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_naive_eval.log \
                     | tee -a /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_naive_scores.txt
done

# Medium difficulty 
for seed in 69 777 888; do
    python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_naive_updated.json \
                     --output /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_naive_eval.json \
                     --log /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_naive_eval.log \
                     | tee -a /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_naive_scores.txt
done

# Hard difficulty
for seed in 69 777 888; do
    python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_naive_updated.json \
                     --output /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_naive_eval.json \
                     --log /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_naive_eval.log \
                     | tee -a /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_naive_scores.txt
done
