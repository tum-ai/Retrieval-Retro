#!/bin/bash

# Hard difficulty
for seed in 777 888 999 42 69; do
    python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/hard_naive/RR_hard_best_results_${seed}_naive_updated_all.json \
                     --output /home/thorben/code/mit/Retrieval-Retro/results/hard_naive/RR_hard_best_results_${seed}_naive_eval_all.pickle \
                     --log /home/thorben/code/mit/Retrieval-Retro/results/hard_naive/RR_hard_best_results_${seed}_naive_eval_all.log \
                     | tee -a /home/thorben/code/mit/Retrieval-Retro/results/hard_naive/RR_hard_best_results_${seed}_naive_scores_all.txt
done

# # Medium difficulty 
# for seed in 42 69 777 888 999; do
#     python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/medium_naive/RR_medium_best_results_${seed}_naive_updated_only_50.json \
#                      --output /home/thorben/code/mit/Retrieval-Retro/results/medium_naive/RR_medium_best_results_${seed}_naive_eval.pickle \
#                      --log /home/thorben/code/mit/Retrieval-Retro/results/medium_naive/RR_medium_best_results_${seed}_naive_eval.log \
#                      | tee -a /home/thorben/code/mit/Retrieval-Retro/results/medium_naive/RR_medium_best_results_${seed}_naive_scores.txt
# done

# # Easy difficulty
# for seed in 42 69 777 888 999; do
#     python eval_3.py --input /home/thorben/code/mit/Retrieval-Retro/results/easy_naive/RR_easy_best_results_${seed}_naive_updated_only_50.json \
#                      --output /home/thorben/code/mit/Retrieval-Retro/results/easy_naive/RR_easy_best_results_${seed}_naive_eval.pickle \
#                      --log /home/thorben/code/mit/Retrieval-Retro/results/easy_naive/RR_easy_best_results_${seed}_naive_eval.log \
#                      | tee -a /home/thorben/code/mit/Retrieval-Retro/results/easy_naive/RR_easy_best_results_${seed}_naive_scores.txt
# done

