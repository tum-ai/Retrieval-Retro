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
     

    save_path = f'./dataset/nre/{difficulty}/year_{mode}_nre_retrieved_{k}_{difficulty}'
    

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

    # precursor_graph = torch.load(f"/home/thorben/code/mit/Retrieval-Retro/dataset/nre/{args.difficulty}/precursor_graphs_{args.difficulty}.pt", map_location=device)
    # precursor_loader = DataLoader(precursor_graph, batch_size = 1, shuffle=False)

    train_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_train.pt')
    valid_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_val.pt')
    test_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_test.pt')

    # train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size = 1)
    # test_loader = DataLoader(test_dataset, batch_size = 1)

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

    # precursor_formation_list = []
    # train_formation_list = []
    # valid_formation_list = []
    # test_formation_list = []

    # model.eval()
    # with torch.no_grad():

    #     for bc, batch in enumerate(precursor_loader):
    #         batch.to(device)

    #         y,_ = model(batch)

    #         precursor_formation_list.append(y)
    #     precursor_y_tensor = torch.stack(precursor_formation_list)
    #     torch.save(precursor_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_precursor_formation_energy.pt')

    #     for bc, batch in enumerate(train_loader):
    #         batch.to(device)

    #         y,_ = model(batch)

    #         train_formation_list.append(y)
    #     train_y_tensor = torch.stack(train_formation_list)
    #     torch.save(train_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_train_formation_energy.pt')

    #     for bc, batch in enumerate(valid_loader):
    #         batch.to(device)

    #         y,_ = model(batch)

    #         valid_formation_list.append(y)
    #     valid_y_tensor = torch.stack(valid_formation_list)
    #     torch.save(valid_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_valid_formation_energy.pt')

    #     for bc, batch in enumerate(test_loader):
    #         batch.to(device)

    #         y,_ = model(batch)

    #         test_formation_list.append(y)
    #     test_y_tensor = torch.stack(test_formation_list)
    #     torch.save(test_y_tensor, f'./dataset/nre/{args.difficulty}/mit_{args.split}_test_formation_energy.pt')
    
    precursor_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_precursor_formation_energy.pt', map_location=device)
    train_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_train_formation_energy.pt', map_location=device)
    valid_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_valid_formation_energy.pt', map_location=device)
    test_formation_y = torch.load(f'./dataset/nre/{args.difficulty}/mit_{args.split}_test_formation_energy.pt', map_location=device)
    K = args.K

    G_train = []
    for i, db in enumerate(train_dataset):
        precursor_indices = db.y_lb_one.nonzero(as_tuple=False).squeeze()
        precursor_energy_sum = precursor_formation_y[precursor_indices].sum()
        G_dic = {
            'precursor_energy_sum': precursor_energy_sum,
            'target_energy': train_formation_y[i],
            'Gibbs_free_energy': train_formation_y[i] - precursor_energy_sum,
            'comp_features': db.comp_fea,
        }
        G_train.append(G_dic)

    with open(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/element_lookup.json', 'r') as f:
        element_lookup = json.load(f)
        common_elements = ['C', 'H', 'O', 'N']
        common_elem_ids= [element_lookup.index(elem) for elem in common_elements]


  # For Train

    train_ref_list = []
    for i, target in tqdm(enumerate(train_dataset)):
        mat_gibbs_dic = {}
        target_mat_ids= target.comp_fea.nonzero(as_tuple=False).squeeze()
        extended_indices = torch.cat([target_mat_ids, torch.tensor(common_elem_ids, device=target_mat_ids.device)])
        for j,ref in enumerate(G_train):
            if i == j: # skip self
                continue
            ref_mat_ids = ref['comp_features'].nonzero(as_tuple=False).squeeze()
            # selecting from those whose precursors contain the same elements as the target material, along with other common elements such as C, H, O, and N.
            if torch.all(torch.isin(target_mat_ids, ref_mat_ids)) and torch.all(torch.isin(ref_mat_ids, extended_indices)):
                mat_gibbs_dic[j] = ref['Gibbs_free_energy']
        train_ref_list.append(mat_gibbs_dic)

    top_k_train_nre_retrieved = {}

    for i, mat_gibbs_list in enumerate(train_ref_list):
        if len(mat_gibbs_list) > 0:
            # Convert dictionary to lists of indices and values
            indices = list(mat_gibbs_list.keys())
            values = list(mat_gibbs_list.values())
            
            # Convert to tensor for topk operation
            values_tensor = torch.tensor(values, device=device)
            
            # Get top K values and their indices
            k = min(K, len(values))
            topk_values, topk_indices = torch.topk(values_tensor, k, largest=False)
            
            # Map back to original indices
            selected_indices = [indices[idx] for idx in topk_indices.tolist()]
            top_k_train_nre_retrieved[i] = selected_indices
        else:
            top_k_train_nre_retrieved[i] = []

    with open(f'./dataset/nre/{args.difficulty}/year_train_nre_retrieved_{K}_marco.json', 'w') as f:
        json.dump(top_k_train_nre_retrieved, f)
    
  
    # For Validation

    valid_ref_list = []
    for i, target in tqdm(enumerate(valid_dataset)):
        mat_gibbs_dic = {}
        target_mat_ids= target.comp_fea.nonzero(as_tuple=False).squeeze()
        extended_indices = torch.cat([target_mat_ids, torch.tensor(common_elem_ids, device=target_mat_ids.device)])
        for j,ref in enumerate(G_train):
            if i == j: # skip self
                continue
            ref_mat_ids = ref['comp_features'].nonzero(as_tuple=False).squeeze()
            # selecting from those whose precursors contain the same elements as the target material, along with other common elements such as C, H, O, and N.
            if torch.all(torch.isin(target_mat_ids, ref_mat_ids)) and torch.all(torch.isin(ref_mat_ids, extended_indices)):
                mat_gibbs_dic[j] = ref['Gibbs_free_energy']
        valid_ref_list.append(mat_gibbs_dic)

    top_k_valid_nre_retrieved = {}

    for i, mat_gibbs_list in enumerate(valid_ref_list):
        if len(mat_gibbs_list) > 0:
            # Convert dictionary to lists of indices and values
            indices = list(mat_gibbs_list.keys())
            values = list(mat_gibbs_list.values())
            
            # Convert to tensor for topk operation
            values_tensor = torch.tensor(values, device=device)
            
            # Get top K values and their indices
            k = min(K, len(values))
            topk_values, topk_indices = torch.topk(values_tensor, k, largest=False)
            
            # Map back to original indices
            selected_indices = [indices[idx] for idx in topk_indices.tolist()]
            top_k_valid_nre_retrieved[i] = selected_indices
        else:
            top_k_valid_nre_retrieved[i] = []

    with open(f'./dataset/nre/{args.difficulty}/year_valid_nre_retrieved_{K}_marco.json', 'w') as f:
        json.dump(top_k_valid_nre_retrieved, f)

    # For Test

    test_ref_list = []
    for i, target in tqdm(enumerate(test_dataset)):
        mat_gibbs_dic = {}
        target_mat_ids= target.comp_fea.nonzero(as_tuple=False).squeeze()
        extended_indices = torch.cat([target_mat_ids, torch.tensor(common_elem_ids, device=target_mat_ids.device)])
        for j,ref in enumerate(G_train):
            if i == j: # skip self
                continue
            ref_mat_ids = ref['comp_features'].nonzero(as_tuple=False).squeeze()
            # selecting from those whose precursors contain the same elements as the target material, along with other common elements such as C, H, O, and N.
            if torch.all(torch.isin(target_mat_ids, ref_mat_ids)) and torch.all(torch.isin(ref_mat_ids, extended_indices)):
                mat_gibbs_dic[j] = ref['Gibbs_free_energy']
        test_ref_list.append(mat_gibbs_dic)

    top_k_test_nre_retrieved = {}

    for i, mat_gibbs_list in enumerate(test_ref_list):
        if len(mat_gibbs_list) > 0:
            # Convert dictionary to lists of indices and values
            indices = list(mat_gibbs_list.keys())
            values = list(mat_gibbs_list.values())
            
            # Convert to tensor for topk operation
            values_tensor = torch.tensor(values, device=device)
            
            # Get top K values and their indices
            k = min(K, len(values))
            topk_values, topk_indices = torch.topk(values_tensor, k, largest=False)
            
            # Map back to original indices
            selected_indices = [indices[idx] for idx in topk_indices.tolist()]
            top_k_test_nre_retrieved[i] = selected_indices
        else:
            top_k_test_nre_retrieved[i] = []

    with open(f'./dataset/nre/{args.difficulty}/year_test_nre_retrieved_{K}_marco.json', 'w') as f:
        json.dump(top_k_test_nre_retrieved, f)




if __name__ == "__main__":

    main()