import torch
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.loader import DataLoader
import json
import torch.nn.functional as F
from collections import defaultdict


def make_sim_mpc(device):

    # Load saved embeddings from trained MPC (Fisrt, you need to train the MPC ,then save the embeddings)
    load_path_train = f'./dataset/train_year_mpc_embeddings.pt'
    load_path_valid = f'./dataset/valid_year_mpc_embeddings.pt'
    load_path_test = f'./dataset/test_year_mpc_embeddings.pt'

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

    torch.save(cos_sim_train, f"./dataset/train_year_mpc_cos_sim_matrix.pt")
    torch.save(cos_sim_valid, f"./dataset/valid_year_mpc_cos_sim_matrix.pt")
    torch.save(cos_sim_test, f"./dataset/test_year_mpc_cos_sim_matrix.pt")

    print(f'cosine similarity matrix mpc saving completed')


def compute_rank_in_batches(tensor, batch_size):

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

def make_retrieved(mode,rank_matrix, k):
     
    save_path = f'./dataset/year_{mode}_mpc_retrieved_{k}'

    candidate_list = defaultdict(list)

    for idx, sim_row in enumerate(rank_matrix):
        top_k_val, top_k_idx = torch.topk(sim_row, k)
        candidate_list[idx] = top_k_idx.tolist()

    with open(save_path, 'w') as f:
            json.dump(candidate_list, f)



def main():


    device = torch.device(f'cuda:{6}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)

    make_sim_mpc(device)

    yr_mpc_train = torch.load(f"./dataset/train_year_mpc_cos_sim_matrix.pt", map_location=device)
    yr_mpc_valid = torch.load(f"./dataset/valid_year_mpc_cos_sim_matrix.pt", map_location=device)
    yr_mpc_test = torch.load(f"./dataset/test_year_mpc_cos_sim_matrix.pt", map_location=device)

    batch_size = 1000
    rank_mpc_train = compute_rank_in_batches(yr_mpc_train, batch_size)

    rank_mpc_valid = compute_rank_in_batches(yr_mpc_valid, batch_size)

    rank_mpc_test = compute_rank_in_batches(yr_mpc_test, batch_size)

    rank_matrix_list = [rank_mpc_train, rank_mpc_valid, rank_mpc_test]


    for idx, matrix in enumerate(rank_matrix_list):

        if idx == 0:
            mode = 'train'

        elif idx == 1:
            mode = 'valid'

        elif idx == 2:
            mode = 'test'

        make_retrieved(mode, matrix, 3)


if __name__ == "__main__":

    main()