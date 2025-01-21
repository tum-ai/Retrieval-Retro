import torch
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.loader import DataLoader
import json

import utils_main



def main(mode, K):
    args = utils_main.parse_args()
    torch.cuda.set_device(args.device)

    train_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_train.pt')
    valid_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_val.pt')
    test_dataset = torch.load(f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/mit_impact_dataset_test.pt')

    save_path = f'/home/thorben/code/mit/Retrieval-Retro/dataset/our_mpc/{args.difficulty}/year_{mode}_mpc_retrieved_{K}'
    save_path_2 = f'/home/thorben/code/mit/Retrieval-Retro/dataset/nre/{args.difficulty}/year_{mode}_nre_final_retrieved_{K}'

    with open(save_path, "r") as f:
        candi_data = json.load(f)

    with open(save_path_2, "r") as f:
        candi_data_2 = json.load(f)

    if mode == "train":
        dataset = train_dataset
        train_dataset = train_dataset

    elif mode == "valid":
        dataset = valid_dataset
        train_dataset = train_dataset

    elif mode == "test":
        dataset = test_dataset
        train_dataset = train_dataset

    new_data = []

    for idx, data in enumerate(dataset):

        tmp = [data]
        candi_idx = candi_data[str(idx)]
        candi_idx_2 = candi_data_2[str(idx)]
        subgraph = []
        subgraph_2 = []
        for i in candi_idx:
            x = train_dataset[i].x
            edge_index = train_dataset[i].edge_index
            fc_weight = train_dataset[i].fc_weight
            edge_attr = train_dataset[i].edge_attr
            comp_fea = train_dataset[i].comp_fea
            y = train_dataset[i].y_lb_one
            candi_graph = Data(x=x, edge_index = edge_index, edge_attr=edge_attr, fc_weight = fc_weight, comp_fea = comp_fea, precursor=y)
            subgraph.append(candi_graph)

        for i in candi_idx_2:
            x = train_dataset[i].x
            edge_index = train_dataset[i].edge_index
            fc_weight = train_dataset[i].fc_weight
            edge_attr = train_dataset[i].edge_attr
            comp_fea = train_dataset[i].comp_fea
            y = train_dataset[i].y_lb_one
            candi_graph2 = Data(x=x, edge_index = edge_index, edge_attr=edge_attr, fc_weight = fc_weight, comp_fea = comp_fea, precursor=y)
            subgraph_2.append(candi_graph2)

        tmp.append(subgraph)
        tmp.append(subgraph_2)
        new_data.append(tuple(tmp))


    torch.save(new_data, f"/home/thorben/code/mit/Retrieval-Retro/dataset/our/{args.difficulty}/year_{mode}_final_mpc_nre_K_{K}.pt")

if __name__ == "__main__":
    from tqdm import tqdm
    mode_list = ['train', 'valid', 'test']
    K_list = [3]
    for mode in tqdm(mode_list):
        for K in K_list:
            main(mode, K)
                
