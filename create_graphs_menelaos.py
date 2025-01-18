import os 

import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch_geometric.data import Data
from pymatgen.core import Composition
import networkx as nx

path = os.getcwd()
df = pd.read_csv('/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/train_data_up_to_2014.csv')
materials = list(df['target_formula'])

df_matscholar = pd.read_csv('/home/thorben/code/mit/Retrieval-Retro/dataset/matscholar.csv', index_col=0)

embeddings_dict = {
    element: torch.tensor(row.values, dtype=torch.float32)
    for element, row in df_matscholar.iterrows()
}

get_unique_targets = []
comp_tuples_list = []
for material in materials:
    fractional_comp = Composition(material).fractional_composition

    sparse_comp_vec = fractional_comp.get_el_amt_dict()
    
    get_comp_tuples = [comp for comp in sparse_comp_vec.items()]
    comp_tuples_list.append(get_comp_tuples)
    
    compounds = list(sparse_comp_vec.keys())
    get_unique_targets.extend(compounds)

lookup = np.unique(np.array(get_unique_targets)).astype(object) #comp_fea attribute
print('lookup', lookup)

data_list = []
for sublist in comp_tuples_list:
    graph = nx.DiGraph()
    lookup_temp = np.copy(lookup)
    x_list = []
    for comp, amount in sublist:
        x_list.append(embeddings_dict[comp])
        # graph.add_node(comp, x=embeddings_dict[comp], target=False)
        lookup_temp[lookup_temp == comp] = amount
    x = torch.stack(x_list)
    # print('x', x)
    comp_fea = torch.tensor([float(x) if isinstance(x, (float, int)) else 0 for x in lookup_temp])
    fc_weight = comp_fea[torch.nonzero(comp_fea)].reshape(-1)

    # nodes = graph.nodes()
    edge_index = []
    edge_attributes = []
    nodes = torch.arange(0,len(x_list))
    for node1 in nodes:
        for node2 in nodes:
            edge_index.append(torch.tensor([node1.item(), node2.item()]))
            # graph.add_edge(node1, node2)
    edge_index = torch.stack(edge_index).reshape(2, -1)
    # edge_attributes = torch.nn.Embedding(num_embeddings=edge_index.shape[1], embedding_dim=400)
    edge_attributes = torch.zeros(edge_index.shape[1], 400)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, fc_weight=fc_weight, comp_fea=comp_fea)
    print(data)
    data_list.append(data)
torch.save(data_list, os.path.join(path, "dataset/mit_impact_dataset.pt"))

