import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from models import MPC
from utils_main import top_k_acc_multiple
import json

def seed_everything(seed):
    """Set seed for reproducibility."""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_model(checkpoint_path, device):
    """Load the trained MPC model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract arguments from the checkpoint
    if 'args' not in checkpoint:
        print("Error: 'args' not found in the checkpoint.")
        sys.exit(1)
    trained_args = checkpoint['args']

    # Initialize model with parameters from the checkpoint
    input_dim = trained_args.input_dim
    hidden_dim = trained_args.hidden_dim
    output_dim = 1309 # Assuming precursor_lookup is a JSON file

    model = MPC(input_dim, hidden_dim, output_dim, device).to(device)
    
    # Load the state dictionary
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"RuntimeError while loading state_dict: {e}")
        sys.exit(1)
    
    model.eval()
    return model, trained_args

def evaluate(model, loader, device, topk=(1,3,5,10)):
    """Evaluate the model and compute top-K accuracies."""
    all_gt_precursors = []
    all_sorted_candidates = []
    all_probabilities = []
    
    topk_correct = {k: 0 for k in topk}
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs, _, _ = model(batch, None)  # Assuming model returns (multi_label, emb, reconstruction)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(len(batch)):
                # Get ground truth labels from y_multiple instead of y_lb_one
                gt_indices = batch.y_multiple[i].nonzero().flatten().cpu().numpy()
                gt_precursors = [model.precursor_lookup[idx] for idx in gt_indices]
                
                probs = probabilities[i]
                sorted_indices = np.argsort(probs)[::-1]
                sorted_candidates = [model.precursor_lookup[idx] for idx in sorted_indices]
                sorted_probs = probs[sorted_indices].tolist()
                
                all_gt_precursors.append(gt_precursors)
                all_sorted_candidates.append(sorted_candidates)
                all_probabilities.append(sorted_probs)
                
                total += len(gt_precursors)
                
                for k in topk:
                    topk_preds = set(sorted_candidates[:k])
                    topk_correct[k] += sum([1 for precursor in gt_precursors if precursor in topk_preds])
    
    accuracies = {f"Top-{k}": (topk_correct[k] / total) if total > 0 else 0 for k in topk}
    
    return accuracies, all_gt_precursors, all_sorted_candidates, all_probabilities

def save_results(results_data, output_path):
    """Save the evaluation results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved evaluation results to {output_path}")

def prepare_results(target_names, gt_precursors, sorted_candidates, probabilities):
    """Prepare the results in the desired format."""
    results = {}
    for name, gt, candidates, probs in zip(target_names, gt_precursors, sorted_candidates, probabilities):
        results[name] = {
            "gt_precursors": gt,
            "sorted_candidates": candidates,
            "probabilities": probs
        }
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate MPC Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'], default='medium',
                        help='Difficulty level of the dataset')
    parser.add_argument('--base_path', type=str, default=os.getcwd(), help='Base path for the project')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to save the evaluation results')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seed_everything(seed=42)
    
    # Load the model and retrieve training args
    model, trained_args = load_model(args.checkpoint, device)
    
    # Define paths based on args.difficulty and args.base_path
    dataset_paths = {
        "easy": {
            "test_dataset": os.path.join(args.base_path, "dataset/our_mpc/easy/mit_impact_dataset_test.pt"),
            "element_lookup": os.path.join(args.base_path, "dataset/our_mpc/easy/element_lookup.json"),
            "precursor_lookup": os.path.join(args.base_path, "dataset/our_mpc/easy/precursor_lookup.json")
        },
        "medium": {
            "test_dataset": os.path.join(args.base_path, "dataset/our_mpc/medium/mit_impact_dataset_test.pt"),
            "element_lookup": os.path.join(args.base_path, "dataset/our_mpc/medium/element_lookup.json"),
            "precursor_lookup": os.path.join(args.base_path, "dataset/our_mpc/medium/precursor_lookup.json")
        },
        "hard": {
            "test_dataset": os.path.join(args.base_path, "dataset/our_mpc/hard/mit_impact_dataset_test.pt"),
            "element_lookup": os.path.join(args.base_path, "dataset/our_mpc/hard/element_lookup.json"),
            "precursor_lookup": os.path.join(args.base_path, "dataset/our_mpc/hard/precursor_lookup.json")
        }
    }

    with open(dataset_paths[args.difficulty]["element_lookup"], 'r') as f:
        element_lookup = json.load(f)
    with open(dataset_paths[args.difficulty]["precursor_lookup"], 'r') as f:
        precursor_lookup = json.load(f)
    
    model.element_lookup = element_lookup
    model.precursor_lookup = precursor_lookup

    # Ensure the selected difficulty exists in dataset_paths
    if args.difficulty not in dataset_paths:
        print(f"Invalid difficulty level: {args.difficulty}. Choose from 'easy', 'medium', 'hard'.")
        sys.exit(1)
    
    selected_paths = dataset_paths[args.difficulty]
    
    # Check if all necessary files exist
    for key, path in selected_paths.items():
        if not os.path.exists(path):
            print(f"Required file for '{key}' not found at: {path}")
            sys.exit(1)
    
    # Load lookups
    with open(selected_paths["element_lookup"], 'r') as f:
        element_lookup = json.load(f)
    with open(selected_paths["precursor_lookup"], 'r') as f:
        precursor_lookup = json.load(f)
    
    # Load test dataset
    test_dataset = torch.load(selected_paths["test_dataset"])
    if len(test_dataset) == 0:
        print(f"The test dataset at {selected_paths['test_dataset']} is empty.")
        sys.exit(1)
        
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    accuracies, gt_precursors, sorted_candidates, probabilities = evaluate(model, test_loader, device)
    
    # Prepare results data
    target_names = [data.y_string_label for data in test_dataset]
    results_formatted = prepare_results(target_names, gt_precursors, sorted_candidates, probabilities)
    results_data = list(results_formatted.items())
    
    # Save results
    save_results(results_data, args.output)
    
    # Print accuracies
    print("Top-K Accuracies:")
    for k, acc in accuracies.items():
        print(f"{k}: {acc:.4f}")

if __name__ == "__main__":
    main()

"""
python evaluate_mpc.py \
       --checkpoint ./checkpoints/mpc/early_stop_model_epoch_240_0.0005_our_data_True.pt \
       --difficulty medium \
       --output medium_evaluation_results.json
"""