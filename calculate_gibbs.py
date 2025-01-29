import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from models import GraphNetwork, GraphNetwork_prop
import utils_main
from collections import defaultdict
from tqdm import tqdm

torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = "4"

def seed_everything(seed):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_retrieved(mode, split, rank_matrix, k, seed, difficulty):
     

    save_path = f'./dataset/nre/{difficulty}/year_{mode}_nre_retrieved_{k}_{difficulty}_naive.json'
    

    candidate_list = defaultdict(list)

    for idx, sim_row in enumerate(rank_matrix):
        top_k_val, top_k_idx = torch.topk(sim_row, k, largest=False)
        candidate_list[idx] = top_k_idx.tolist()

    with open(save_path, 'w') as f:
            json.dump(candidate_list, f)

def main():
    args = utils_main.parse_args()
    train_config = utils_main.training_config(args)
    # configuration = utils_nre.exp_get_name1(train_config)
    # print(f'configuration: {configuration}')

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)
    seed_everything(seed=args.seed)

    precursor_graph = torch.load(f"/home/thorben/code/mit/Retrieval-Retro/dataset/nre/{args.difficulty}/precursor_graphs_{args.difficulty}.pt", map_location=device)
    precursor_loader = DataLoader(precursor_graph, batch_size = 1, shuffle=False)

    train_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_train.pt')
    valid_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_val.pt')
    test_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_test.pt')

    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size = 1)
    test_loader = DataLoader(test_dataset, batch_size = 1)

    print("Dataset Loaded!")


    n_hidden = args.hidden
    n_atom_feat = train_dataset[0].x.shape[1]
    n_bond_feat = train_dataset[0].edge_attr.shape[1]
    output_dim = train_dataset[0].y_multiple.shape[1] #Dataset precursor set dim 

    model = GraphNetwork_prop(args.layers, n_atom_feat, n_bond_feat, n_hidden, device).to(device)
    checkpoint = torch.load("/home/thorben/code/mit/Retrieval-Retro/checkpoints/nre/mit_nre_finetune_experimental_formation_energy/model_best_mae_0.0759.pt", map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f'\nModel Weight Loaded')


    # ### Calculating formation energy for precursor graph ###

    precursor_formation_list = []
    train_formation_list = []
    valid_formation_list = []
    test_formation_list = []

    model.eval()
    with torch.no_grad():

        for bc, batch in enumerate(precursor_loader):
            batch.to(device)

            y,_ = model(batch)

            precursor_formation_list.append(y)
        precursor_y_tensor = torch.stack(precursor_formation_list)
        torch.save(precursor_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_precursor_formation_energy_naive.pt')

        for bc, batch in enumerate(train_loader):
            batch.to(device)

            y,_ = model(batch)

            train_formation_list.append(y)
        train_y_tensor = torch.stack(train_formation_list)
        torch.save(train_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_train_formation_energy_naive.pt')

        for bc, batch in enumerate(valid_loader):
            batch.to(device)

            y,_ = model(batch)

            valid_formation_list.append(y)
        valid_y_tensor = torch.stack(valid_formation_list)
        torch.save(valid_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_valid_formation_energy_naive.pt')

        for bc, batch in enumerate(test_loader):
            batch.to(device)

            y,_ = model(batch)

            test_formation_list.append(y)
        test_y_tensor = torch.stack(test_formation_list)
        torch.save(test_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_test_formation_energy_naive.pt')
    
    precursor_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_precursor_formation_energy_naive.pt', map_location=device)
    train_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_train_formation_energy_naive.pt', map_location=device)
    valid_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_valid_formation_energy_naive.pt', map_location=device)
    test_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_test_formation_energy_naive.pt', map_location=device)
    K = args.K

    train_energies = []
    for i, db in enumerate(train_dataset):
        precursor_indices = db.y_lb_one.nonzero(as_tuple=False).squeeze()
        precursor_energies = precursor_formation_y[precursor_indices].sum()
        train_energies.append(precursor_energies)
    
    train_energies = torch.tensor(train_energies).to(device)

    
    # For Train

    train_idx = []
    # Compute formation energy differences
    for data in tqdm(train_formation_y):
        precursor_differences = data - train_energies   
        train_idx.append(precursor_differences)

    # Process in chunks of 500 to save memory
    chunk_size = 5
    num_chunks = (len(train_formation_y) + chunk_size - 1) // chunk_size
    train_matrix = None
    # Process chunks
    for i in tqdm(range(num_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(train_idx))
        chunk = torch.cat(train_idx[start_idx:end_idx], dim=0)
        # Add diagonal elements only for this chunk
        diag_mask = torch.zeros(end_idx - start_idx, len(train_dataset), device=device)
        for j in range(end_idx - start_idx):
            if start_idx + j < len(train_dataset):
                diag_mask[j, start_idx + j] = 100000
        chunk = chunk + diag_mask
        if train_matrix is None:
            train_matrix = chunk
        else:
            train_matrix = torch.cat((train_matrix, chunk), dim=0)

    torch.save(train_matrix, f'./dataset/nre/{args.difficulty}/optimized_{args.split}_train_formation_energy_calculation_delta_G_naive.pt')
    make_retrieved('train','year', train_matrix, K, 0, args.difficulty)

    # For Valid
    valid_idx = []

    # Compute formation energy differences
    for data in tqdm(valid_formation_y):
        precursor_differences = data - train_energies
        valid_idx.append(precursor_differences)

    # Process valid in chunks
    chunk_size = 5
    num_chunks = (len(valid_formation_y) + chunk_size - 1) // chunk_size
    valid_matrix = None
    # Process chunks
    for i in tqdm(range(num_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(valid_idx))
        chunk = torch.cat(valid_idx[start_idx:end_idx], dim=0)
        # Add diagonal elements only for this chunk
        diag_mask = torch.zeros(end_idx - start_idx, len(train_dataset), device=device)
        for j in range(end_idx - start_idx):
            if start_idx + j < len(train_dataset):
                diag_mask[j, start_idx + j] = 100000
        chunk = chunk + diag_mask
        if valid_matrix is None:
            valid_matrix = chunk
        else:
            valid_matrix = torch.cat((valid_matrix, chunk), dim=0)

    torch.save(valid_matrix, f'./dataset/nre/{args.difficulty}/optimized_{args.split}_valid_formation_energy_calculation_delta_G_naive.pt')
    make_retrieved('valid','year', valid_matrix, K, 0, args.difficulty)

    # For Test
    test_idx = []

    # Compute formation energy differences
    for data in tqdm(test_formation_y):
        precursor_differences = data - train_energies
        test_idx.append(precursor_differences)

    # Process test in chunks
    chunk_size = 5
    num_chunks = (len(test_formation_y) + chunk_size - 1) // chunk_size
    test_matrix = None
    # Process chunks  
    for i in tqdm(range(num_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(test_idx))
        chunk = torch.cat(test_idx[start_idx:end_idx], dim=0)
        # Add diagonal elements only for this chunk
        diag_mask = torch.zeros(end_idx - start_idx, len(train_dataset), device=device)
        for j in range(end_idx - start_idx):
            if start_idx + j < len(train_dataset):
                diag_mask[j, start_idx + j] = 100000
        chunk = chunk + diag_mask
        if test_matrix is None:
            test_matrix = chunk
        else:
            test_matrix = torch.cat((test_matrix, chunk), dim=0)

    # Stack the differences and add a large value to the diagonal
    torch.save(test_matrix, f'./dataset/nre/{args.difficulty}/optimized_{args.split}_test_formation_energy_calculation_delta_G_naive.pt')
    make_retrieved('test','year', test_matrix, K, 0, args.difficulty)



if __name__ == "__main__":

    main()