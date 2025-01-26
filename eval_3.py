import argparse
import json
import itertools
import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Composition
import datetime
import pickle

class PrecursorEvaluator:
    """
    A class to evaluate precursor candidates against a target composition 
    and find valid combinations of a given size, then rank them by 
    probability product (faster version).
    """
    def __init__(self, target_comp, precursors, gt_precursors):
        """
        Parameters
        ----------
        target_comp : Composition
            The target composition (e.g., Composition('Ba4 Sm2 Ti3 Nb6.5 Fe0.5 O30')).
        precursors : list of (Composition, float)
            List of tuples of the form (Composition, probability).
        gt_precursors : list of Composition
            The known ground-truth precursor compositions.
        """
        self.target_comp = target_comp

        # Store precursor data: (composition, probability, set_of_elements)
        self.precursors = []
        for comp, prob in precursors:
            elem_set = set(comp.keys())  # all element symbols
            self.precursors.append((comp, prob, elem_set))

        # Convert ground-truth precursors to a set of Composition objects
        self.gt_precursors_set = set(gt_precursors)

        # Compute ground-truth probability product once
        self.gt_prob_product = 1.0
        for comp in gt_precursors:
            # We must find its probability from self.precursors
            # Make a dictionary if needed. For demonstration, we do a simple loop:
            found = False
            for c2, p2, _ in self.precursors:
                if c2 == comp:
                    self.gt_prob_product *= p2
                    found = True
                    break
            if not found:
                # If a GT precursor wasn't found in our pruned list, prob = 0
                self.gt_prob_product = 0
                break

    def evaluate(self, ground_truth_size, top_n=5):
        """
        Enumerate all precursor combinations of size `ground_truth_size`, 
        check if their combined elements cover the target's elements,
        and record the top_n by probability product. Also find the rank of 
        the ground-truth set if it appears in the combos.

        Parameters
        ----------
        ground_truth_size : int
            Combination size (e.g. number of precursors in ground truth).
        top_n : int
            How many top combos to store/print.

        Returns
        -------
        (rank, top_combos) : (int or None, list of (prob_product, set_of_Compositions))
            rank: The rank of the ground-truth set or None if not found.
            top_combos: The top_n combos in descending order by probability product.
        """

        # Prepare a min-heap for the top combos
        top_heap = []
        heapq.heapify(top_heap)

        # For rank calculation
        count_higher_than_gt = 0
        found_gt = False

        # Quick set for target elements
        elements_target = set(self.target_comp.keys())

        # We will iterate over indices, so let's extract for speed
        # (comp, prob, elem_set)
        n_prec = len(self.precursors)

        # Precompute a dictionary from Composition -> index (for ground-truth check)
        comp_to_idx = { self.precursors[i][0]: i for i in range(n_prec) }

        # Convert ground-truth compositions to a set of indices
        gt_indices = set()
        for gt_comp in self.gt_precursors_set:
            if gt_comp in comp_to_idx:
                gt_indices.add(comp_to_idx[gt_comp])

        # If some GT composition wasn't found, ground-truth set might never appear
        # but we can still keep going.
        all_valid_combos = []
        combo_probs = []
        # Enumerate combos by their indices
        for combo_indices in itertools.combinations(range(n_prec), ground_truth_size):
            # Union of elements
            union_set = set()
            prob_product = 1.0

            # Also track the composition objects for ground-truth check, if needed
            combo_comps = []

            for idx in combo_indices:
                # The set of elements for this precursor
                union_set |= self.precursors[idx][2]
                # Multiply prob
                prob_product *= self.precursors[idx][1]
                combo_comps.append(self.precursors[idx][0])

                # Optional: Early prune if prob_product is already < some threshold
                #   if prob_product < 1e-10: break

            # Check if target elements are covered
            if not elements_target.issubset(union_set):
                continue

            # We have a valid combo - update rank info
            if prob_product > self.gt_prob_product:
                count_higher_than_gt += 1

            # Check if this combo is exactly the ground-truth set
            # Convert to set of indices or set of Compositions
            combo_set = set(combo_comps)
            all_valid_combos.append([x.formula.replace(' ', '') for x in list(combo_set)])
            combo_probs.append(prob_product)
            if combo_set == self.gt_precursors_set:
                found_gt = True

            # Maintain the top_n combos in a min-heap
            # Python's heapq is a min-heap, so we push negative prob for "max" ordering
            if len(top_heap) < top_n:
                heapq.heappush(top_heap, (prob_product, combo_set))
            else:
                # If better than the smallest in the heap, replace
                if prob_product > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (prob_product, combo_set))

        # After enumerating all combos:
        if found_gt:
            rank = count_higher_than_gt + 1
        else:
            rank = None

        # Sort the heap (it has only top_n combos) in descending order
        top_heap.sort(key=lambda x: x[0], reverse=True)
        top_combos = [(prob, combo_set) for (prob, combo_set) in top_heap]

        return rank, top_combos, all_valid_combos, combo_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate precursor candidates for target compositions.")
    
    # Input and output paths
    parser.add_argument("--input", default="results/hard/RR_hard_best_results_777_updated_only_50.json",help="Path to the input JSON file containing target compositions and candidates.")
    parser.add_argument("--output", default="results.pkl", help="Path to save the evaluation results (pickle format).")
    parser.add_argument("--log", default="evaluation_log.txt", help="Path to save the evaluation log.")

    # Filtering and evaluation parameters
    parser.add_argument("--prob-threshold", type=float, default=0.5, help="Probability threshold for candidate filtering.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top combinations to record.")
    parser.add_argument("--k-values", type=int, nargs='+', default=[1, 3, 5, 10, 20, 30, 40], help="Values of K for precision-at-K evaluation.")
    
    args = parser.parse_args()

    # Load input JSON data
    with open(args.input, 'r') as file:
        data = json.load(file)
    file_name = args.input.split('/')[-1]

    results = []
    valid_ranks = []

    for item in tqdm(data):
        # Construct the target composition
        target_mat = str(list(item.keys())[0])
        anchor = Composition(target_mat)
        res_i = {}

        # Ground-truth precursors
        gt_precursors = [Composition(x) for x in item[target_mat]["gt_precursors"]]
        dict_gt_precursors = [x.formula.replace(' ', '') for x in gt_precursors]

        # Candidate compositions and probabilities
        candidates = [Composition(x) for x in item[target_mat]["sorted_candidates"][:30]]
        probabilities = item[target_mat]["sorted_probabilities"][:30]

        # Filter candidates based on probability threshold
        prec_list = [
            (comp, prob)
            for comp, prob in zip(candidates, probabilities)
            # if prob >= args.prob_threshold
        ]

        prec_list = prec_list

        # Create evaluator
        evaluator = PrecursorEvaluator(
            target_comp=anchor,
            precursors=prec_list,
            gt_precursors=gt_precursors
        )

        # Evaluate
        size = len(gt_precursors)
        rank, top_combos, all_valid_combos, combo_probs = evaluator.evaluate(size, top_n=args.top_n)
        
        if rank is not None:
            valid_ranks.append(rank)
        
        res_i["all_combos"] = list(zip(all_valid_combos, combo_probs))
        res_i["gt_precursors"] = (dict_gt_precursors, evaluator.gt_prob_product)
        res_i["candidates"] = [x.formula.replace(' ', '') for x in candidates]
        res_i["sorted_probabilities"] = probabilities
        res_i["rank"] = rank
        results.append(res_i)

        # Log results
        with open(args.log, 'a') as log_file:
            log_file.write(f"\nTarget: {target_mat}\n")
            log_file.write(f"Ground-truth probability product: {evaluator.gt_prob_product:.4g}\n")
            log_file.write(f"Rank of ground-truth set: {rank}\n")
            log_file.write(f"Ground Truth Precursors: {[c.formula for c in gt_precursors]}\n")
            log_file.write("\nTop combos (prob, compositions):\n")
            for i, (prob, combo_set) in enumerate(top_combos):
                combo_str = [c.formula for c in combo_set]
                log_file.write(f"{i+1}  - prob={prob:.4g}, combo={combo_str}\n")
            log_file.write("-" * 40)

    # Compute and display summary statistics
    avg_rank = np.nanmean(valid_ranks) if valid_ranks else None
    num_nones = len(results) - len(valid_ranks)
    print(f"Ranks that are not None: {len(valid_ranks)}")
    print(f"Average rank: {avg_rank}")
    print(f"Number of None in ranks: {num_nones}")
    
    for k in args.k_values:
        valid_ranks_up_to_k = sum(1 for r in valid_ranks if r is not None and r <= k)
        prec_at_k = valid_ranks_up_to_k / len(results)
        print(f"Precision at K={k}: {prec_at_k:.4f}")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval_res_{timestamp}_{file_name}_.pkl"
    with open(args.output, 'wb') as outfile:
        pickle.dump(results, outfile)
    print(f"Results saved to {args.output}")
