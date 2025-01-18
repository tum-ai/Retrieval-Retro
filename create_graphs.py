import os 
import ast

import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch_geometric.data import Data
from pymatgen.core import Composition

def get_unique_elements(csv_path):
    """Extract unique elements from a dataset CSV file."""
    df = pd.read_csv(csv_path)
    materials = list(df['target_formula'])
    precursors = list(df['precursor_formulas'])

    unique_elements = set()
    for material in materials:
        try:
            fractional_comp = Composition(material).element_composition
            elements = fractional_comp.get_el_amt_dict().keys()
            unique_elements.update(elements)
        except Exception as e:
            print(f"Error processing material {material}: {e}")

    all_prec = []
    for precursor_set in precursors: 
        formulas = ast.literal_eval(precursor_set)
        for formula in formulas:
            all_prec.extend(formula)
    unique_pre = np.unique(np.array(all_prec))
   
    return unique_elements, unique_pre

def create_lookup_from_all_datasets(train_path, val_path, test_path):
    """Create a consistent lookup table from all datasets."""
    all_elements = set()
    for path in [train_path, val_path, test_path]:
        if path and os.path.exists(path):
            elements = get_unique_elements(path)
            all_elements.update(elements)
    
    # Convert to sorted list for consistency
    lookup = np.array(sorted(all_elements))
    return lookup

def create_graph_dataset(input_csv_path, matscholar_path, dataset_type, lookup):
    """
    Create graph dataset from input CSV file and save as PyTorch geometric dataset
    
    Args:
        input_csv_path (str): Path to input CSV file containing material formulas
        matscholar_path (str): Path to matscholar embeddings CSV file
        dataset_type (str): Type of dataset, either 'train', 'val', or 'test'
    """
    
    materials, unique_precursors = get_unique_elements('/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/train_data_up_to_2014.csv')
    
    # Read input data
    df = pd.read_csv(input_csv_path)
    
    df_matscholar = pd.read_csv(matscholar_path, index_col=0)

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
        
    data_list = []
    for sublist, precursor_set in zip(comp_tuples_list, unique_precursors):
        lookup_temp = np.copy(lookup)
        x_list = []
        for comp, amount in sublist:
            x_list.append(embeddings_dict[comp])
            lookup_temp[lookup_temp == comp] = amount
        x = torch.stack(x_list)
        
        comp_fea = torch.tensor([float(x) if isinstance(x, (float, int)) else 0 for x in lookup_temp], dtype=torch.float32)
        fc_weight = comp_fea[torch.nonzero(comp_fea)].reshape(-1).to(torch.float32)

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
        
        
        # TODO: add y_multiple, y_multiple_len, y_lb_freq, y_lb_avg, y_lb_all
        # y_multiple is output dimension of the model
        y_multiple = torch.zeros(1, 798)

        #
        #y_multiple_len = torch.zeros(1)
        #y_lb_avg = torch.zeros(1)
        #y_lb_all = torch.zeros(1)
        #y_lb_one = torch.zeros(1)

        
        # train Data(x=[3, 200], edge_index=[2, 9], edge_attr=[9, 400], fc_weight=[3], comp_fea=[83], y_exp_form=-0.8730636239051819)
        # val Data(x=[3, 200], edge_index=[2, 9], edge_attr=[9, 400], fc_weight=[3], comp_fea=[83], y_multiple=[1, 798], y_multiple_len=1, y_lb_freq=[798], y_lb_avg=[798], y_lb_all=[798], y_lb_one=[798])
        # test Data(x=[3, 200], edge_index=[2, 9], edge_attr=[9, 400], fc_weight=[3], comp_fea=[83], y_multiple=[1, 798], y_multiple_len=1, y_lb_freq=[798], y_lb_avg=[798], y_lb_all=[798], y_lb_one=[798])
        
        if dataset_type == 'train':
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, fc_weight=fc_weight, comp_fea=comp_fea, y_multiple=y_multiple)
        elif dataset_type == 'val' or dataset_type == 'test':
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, fc_weight=fc_weight, comp_fea=comp_fea, y_multiple=y_multiple, y_multiple_len=y_multiple_len, y_lb_freq=y_lb_freq, y_lb_avg=y_lb_avg, y_lb_all=y_lb_all, y_lb_one=y_lb_one)
        # print(data)
        data_list.append(data)
    torch.save(data_list, os.path.join(path, "dataset/mit_impact_dataset_" + dataset_type + ".pt"))
    return data_list

if __name__ == "__main__":
    # Example usage
    path = os.getcwd()
    matscholar_path = '/home/thorben/code/mit/Retrieval-Retro/dataset/matscholar.csv'

    # Create train dataset
    train_input = '/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/train_data_up_to_2014.csv'
    val_input = '/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/val_data_up_to_2014.csv'
    test_input = '/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/test_data_after_2014.csv'
    train_output = os.path.join(path, "dataset/mit_impact_dataset_train.pt")
    val_output = os.path.join(path, "dataset/mit_impact_dataset_val.pt")
    test_output = os.path.join(path, "dataset/mit_impact_dataset_test.pt")

    lookup = create_lookup_from_all_datasets(train_input, val_input, test_input)

    train_data = create_graph_dataset(train_input, matscholar_path, 'train', lookup)
    #val_data = create_graph_dataset(val_input, matscholar_path, 'val', lookup)
    #test_data = create_graph_dataset(test_input, matscholar_path, 'test', lookup)

    # print first 10 elements of train_data
    for i in range(10):
        print(train_data[i])




