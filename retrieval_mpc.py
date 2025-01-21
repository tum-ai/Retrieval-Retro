import torch
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.loader import DataLoader
import json
import torch.nn.functional as F
from collections import defaultdict
import os
from models import MPC, MultiLossLayer, CircleLoss
import utils_main
def make_sim_mpc(device, difficulty):

    # Load saved embeddings from trained MPC (Fisrt, you need to train the MPC ,then save the embeddings)
    load_path_train = f'./dataset/our_mpc/{difficulty}/mit_mpc_train_year_embeddings.pt'
    load_path_valid = f'./dataset/our_mpc/{difficulty}/mit_mpc_valid_year_embeddings.pt'
    load_path_test = f'./dataset/our_mpc/{difficulty}/mit_mpc_test_year_embeddings.pt'

    train_emb = torch.load(load_path_train, map_location = device).squeeze(1)
    valid_emb = torch.load(load_path_valid, map_location = device).squeeze(1)
    test_emb = torch.load(load_path_test, map_location = device).squeeze(1)

    train_emb_norm = F.normalize(train_emb, p=2, dim=1)
    valid_emb_norm = F.normalize(valid_emb, p=2, dim=1)
    test_emb_norm = F.normalize(test_emb, p=2, dim=1)

    cos_sim_train = torch.mm(train_emb_norm, train_emb_norm.t())
    cos_sim_valid = torch.mm(valid_emb_norm, train_emb_norm.t())
    cos_sim_test = torch.mm(test_emb_norm, train_emb_norm.t())

    diag_mask = torch.ones_like(cos_sim_train).to(device) - torch.eye(cos_sim_train.size(0), dtype=torch.float32).to(device)
    cos_sim_train= cos_sim_train * diag_mask

    torch.save(cos_sim_train, f"./dataset/our_mpc/{difficulty}/train_year_mpc_cos_sim_matrix.pt")
    torch.save(cos_sim_valid, f"./dataset/our_mpc/{difficulty}/valid_year_mpc_cos_sim_matrix.pt")
    torch.save(cos_sim_test, f"./dataset/our_mpc/{difficulty}/test_year_mpc_cos_sim_matrix.pt")

    print(f'cosine similarity matrix mpc saving completed')


def compute_rank_in_batches(tensor, batch_size, difficulty):

    # Zeros for the ranked tensor
    ranked_tensor = torch.zeros_like(tensor)
    
    # Compute the number of batches
    num_batches = tensor.size(0) // batch_size + (1 if tensor.size(0) % batch_size else 0)
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, tensor.size(0))
        batch = tensor[batch_start:batch_end]
        # Perform the ranking operation on the smaller batch
        batch_ranked = batch.argsort(dim=1).argsort(dim=1)
        ranked_tensor[batch_start:batch_end] = batch_ranked
    
    return ranked_tensor

def make_retrieved(mode,rank_matrix, k, difficulty):
     
    save_path = f'./dataset/our_mpc/{difficulty}/year_{mode}_mpc_retrieved_{k}'

    candidate_list = defaultdict(list)

    for idx, sim_row in enumerate(rank_matrix):
        top_k_val, top_k_idx = torch.topk(sim_row, k)
        candidate_list[idx] = top_k_idx.tolist()

    with open(save_path, 'w') as f:
            json.dump(candidate_list, f)


def extract_embeddings(model_path, device, difficulty):
    """Extract embeddings using the saved model."""
    # Load datasets
    train_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{difficulty}/mit_impact_dataset_train.pt')
    valid_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{difficulty}/mit_impact_dataset_val.pt')
    test_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{difficulty}/mit_impact_dataset_test.pt')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint['args']
    
    # Get dimensions from args
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = train_dataset[0].y_multiple.shape[1]  # Same as in training
    
    print(f"Model dimensions from args - input: {input_dim}, hidden: {hidden_dim}, output: {output_dim}")
    
    # Initialize model with correct dimensions
    model = MPC(input_dim, hidden_dim, output_dim, device).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract embeddings
    embeddings = {}
    for name, loader in [('train', train_loader), ('valid', valid_loader), ('test', test_loader)]:
        all_emb = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, emb, _ = model(batch, None)
                all_emb.append(emb)
        embeddings[name] = torch.cat(all_emb, dim=0)
        torch.save(embeddings[name], f'./dataset/our_mpc/{difficulty}/mit_mpc_{name}_year_embeddings.pt')
        print(f"Saved {name} embeddings with shape: {embeddings[name].shape}")
    
    return embeddings

def main():

    args = utils_main.parse_args()
    # Check available GPUs
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device = torch.device(f'cuda:{n_gpu-1}' if n_gpu > 0 else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Extract embeddings first
    model_path = "/home/thorben/code/mit/Retrieval-Retro/checkpoints/mpc/early_stop_model_epoch_290_0.0005_our_data_hard.pt"
    embeddings = extract_embeddings(model_path, device, args.difficulty)
    
    # Then continue with similarity computation
    make_sim_mpc(device, args.difficulty)

    yr_mpc_train = torch.load(f"./dataset/our_mpc/{args.difficulty}/train_year_mpc_cos_sim_matrix.pt", map_location=device)
    yr_mpc_valid = torch.load(f"./dataset/our_mpc/{args.difficulty}/valid_year_mpc_cos_sim_matrix.pt", map_location=device)
    yr_mpc_test = torch.load(f"./dataset/our_mpc/{args.difficulty}/test_year_mpc_cos_sim_matrix.pt", map_location=device)

    batch_size = 1000
    rank_mpc_train = compute_rank_in_batches(yr_mpc_train, batch_size, args.difficulty)

    rank_mpc_valid = compute_rank_in_batches(yr_mpc_valid, batch_size, args.difficulty)

    rank_mpc_test = compute_rank_in_batches(yr_mpc_test, batch_size, args.difficulty)

    rank_matrix_list = [rank_mpc_train, rank_mpc_valid, rank_mpc_test]


    for idx, matrix in enumerate(rank_matrix_list):

        if idx == 0:
            mode = 'train'

        elif idx == 1:
            mode = 'valid'

        elif idx == 2:
            mode = 'test'

        make_retrieved(mode, matrix, 3, args.difficulty)


if __name__ == "__main__":

    main()