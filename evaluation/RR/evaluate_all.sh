#!/bin/bash

# Easy difficulty
# Easy difficulty
for seed in 69 777 888; do
    python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_updated.json \
                     --output /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_eval.json \
                     --log /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_eval.log \
                     | tee -a /home/thorben/code/mit/Retrieval-Retro/results/easy/RR_easy_best_results_${seed}_scores.txt
done

# Medium difficulty 
for seed in 69 777 888; do
    python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_updated.json \
                     --output /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_eval.json \
                     --log /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_eval.log \
                     | tee -a /home/thorben/code/mit/Retrieval-Retro/results/medium/RR_medium_best_results_${seed}_scores.txt
done

# Hard difficulty
for seed in 69 777 888; do
    python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_updated.json \
                     --output /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_eval.json \
                     --log /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_eval.log \
                     | tee -a /home/thorben/code/mit/Retrieval-Retro/results/hard/RR_hard_best_results_${seed}_scores.txt
done
