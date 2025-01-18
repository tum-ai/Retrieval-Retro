import os 
import ast 

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
precursors = list(df['precursor_formulas'])

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

all_prec = []
for precursor_set in precursors: 
    formulas = ast.literal_eval(precursor_set)
    for formula in formulas:
        all_prec.append(formula)
unique_pre = np.unique(all_prec)

data_list = []
for sublist, precursor_set in zip(comp_tuples_list, precursors):
    lookup_temp = np.copy(lookup)
    x_list = []
    for comp, amount in sublist:
        x_list.append(embeddings_dict[comp])
        lookup_temp[lookup_temp == comp] = amount
    x = torch.stack(x_list)
    
    comp_fea = torch.tensor([float(x) if isinstance(x, (float, int)) else 0 for x in lookup_temp])
    fc_weight = comp_fea[torch.nonzero(comp_fea)].reshape(-1)

    precursor_set = ast.literal_eval(precursor_set)    
    y_lb_freq = torch.zeros(unique_pre.shape)
    for precursor in precursor_set: 
        y_lb_freq[np.where(precursor==unique_pre)] = 1
    y_multiple = y_lb_freq.reshape(1, -1)

    edge_index = []
    edge_attributes = []
    nodes = torch.arange(0,len(x_list))
    for node1 in nodes:
        for node2 in nodes:
            edge_index.append(torch.tensor([node1.item(), node2.item()]))
    edge_index = torch.stack(edge_index).reshape(2, -1)
    
    edge_attributes = torch.rand(edge_index.shape[1], 400) #TODO add random init
    
    data = Data(x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attributes, 
                fc_weight=fc_weight, 
                comp_fea=comp_fea, 
                y_lb_freq=y_lb_freq, 
                y_multiple=y_multiple, 
                y_lb_avg=y_lb_freq, 
                y_lb_all=y_lb_freq, 
                y_lb_one=y_lb_freq)
    
    data_list.append(data)
torch.save(data_list, os.path.join(path, "dataset/mit_impact_dataset.pt"))

