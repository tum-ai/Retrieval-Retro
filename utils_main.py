
import torch
from itertools import groupby
from sklearn.metrics import recall_score, precision_score, f1_score
import argparse
import numpy as np
from itertools import combinations



def product(iterable, prob):
    result = 1
    for element in iterable:
        result *= prob[element]
    return result

def find_top_k_product_sets(arr, prob, subset_size, k):
    max_sets = []
    max_products = []
    all_subsets = list(combinations(arr, subset_size))
    product_values = [product(subset, prob) for subset in all_subsets]
    max_indices = np.argsort(product_values)[-k:][::-1]

    max_sets = [all_subsets[i] for i in max_indices]
    max_products = [product_values[i] for i in max_indices]

    return max_sets, max_products

def recall_multilabel_multiple(y_true_matrix, y_pred_prob, threshold=None): 
    num_products = 10
    sorted_indices = np.argsort(y_pred_prob, axis=1)[:, ::-1]
    micro = []
    macro = []
    num_set = len(np.where(y_pred_prob > 0.5)[0])

    if len(y_true_matrix[0]) == 1:
        macro_tmp = []
        true_labels = np.where(y_true_matrix[0].detach().cpu().numpy() ==1)[1]
        result_sets, result_products = find_top_k_product_sets(sorted_indices[0][:num_products], y_pred_prob[0], num_set, 1)
        result_sets = [tuple(sorted(t)) for t in result_sets]

        if tuple(true_labels) in result_sets: 
            micro.append(1)
            macro_tmp.append(1)
        else: 
            micro.append(0)
            macro_tmp.append(0)

        macro.append(np.mean(np.array(macro_tmp)))

    else:
        result_sets, result_products = find_top_k_product_sets(sorted_indices[0][:num_products], y_pred_prob[0], num_set, len(y_true_matrix[0]))
        result_sets = [tuple(sorted(t)) for t in result_sets]
        macro_tmp = []
        for idx, y_true in enumerate(y_true_matrix[0]):

            true_labels = np.where(y_true.detach().cpu().numpy() ==1)[0]       
            
            if tuple(true_labels) in result_sets:
                micro.append(1)
                macro_tmp.append(1)
            else:
                micro.append(0)
                macro_tmp.append(0)

            macro.append(np.mean(np.array(macro_tmp)))

    return np.array(micro), np.array(macro)  



def top_k_acc_multiple(y_true_matrix, y_pred_prob, k): 
    num_examples = len(y_true_matrix)
    num_products = 10

    sorted_indices = np.argsort(y_pred_prob, axis=1)[:, ::-1]

    num_set = len(np.where(y_pred_prob > 0.5)[0])
    idx = 0 
    result_sets, result_products = find_top_k_product_sets(sorted_indices[idx][:num_products], y_pred_prob[idx], num_set, k)
    result_sets = [tuple(sorted(t)) for t in result_sets]

    top_k = []

    for idx, y_true in enumerate(y_true_matrix):
        
        true_labels = np.where(y_true.detach().cpu().numpy() ==1)[0]

        if tuple(true_labels) in result_sets:
            top_k.append(1)
            break

        if idx == (num_examples-1):
            top_k.append(0)


    return np.array(top_k)

# LOG_EPSILON = 1e-5

# def neg_log(x):
#     return - torch.log(x + LOG_EPSILON)

# def log_loss(preds, targs):
#     return targs * neg_log(preds)

# def log_normal(x, m, v):
#     log_prob = (-0.5 * (torch.log(v) + (x-m).pow(2) / v)).sum(-1)
#     return log_prob

# def log_normal_mixture(z, m, v, mask=None):
#     m = m.unsqueeze(0).expand(z.shape[0], -1, -1)
#     v = v.unsqueeze(0).expand(z.shape[0], -1, -1)
#     batch, mix, dim = m.size()
#     z = z.view(batch, 1, dim).expand(batch, mix, dim)
#     indiv_log_prob = log_normal(z, m, v) + torch.ones_like(mask)*(-1e6)*(1.-mask)
#     log_prob = log_mean_exp(indiv_log_prob, mask)
#     return log_prob

# def log_mean_exp(x, mask):
#     return log_sum_exp(x, mask) - torch.log(mask.sum(1))

# def log_sum_exp(x, mask):
#     max_x = torch.max(x, 1)[0]
#     new_x = x - max_x.unsqueeze(1).expand_as(x)
#     return max_x + (new_x.exp().sum(1)).log()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", '-d', type=int, default=6, help="GPU to use")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of Epochs for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size") 
    parser.add_argument("--layers", "-l", type=int, default=3, help="The number layers of the Processor")
    parser.add_argument("--t_layers", type=int, default=2, help="The number of Transformer layers")
    parser.add_argument("--t_layers_sa", type=int, default=1, help="The number of Transformer layers for SA")
    parser.add_argument("--head", type=int, default=1, help="The number of attetion in Transformer ")
    parser.add_argument("--eval", type=int, default=5, help="evaluation step")
    parser.add_argument("--es", type=int, default=30, help="Early Stopping Criteria")
    parser.add_argument("--embedder", type=str, default="RetroPLEX", help="Early Stopping Criteria")
    parser.add_argument("--input_dim", type=int, default=200, help="Early Stopping Criteria")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Early Stopping Criteria")
    parser.add_argument("--output_dim", type=int, default=256, help="Early Stopping Criteria")
    parser.add_argument("--gnn", type=str, default="GraphNetwork", help="gnn models")
    parser.add_argument("--seed", type=int, default=0, help = 'Random seed')
    parser.add_argument("--num_heads", type=int, default=1, help = 'Number of MHA')
    parser.add_argument("--split", type=str, default = 'year', help = 'Dataset Split Strategy')
    parser.add_argument("--lb", type=str, default = 'one', help = 'Label Strategy')
    parser.add_argument("--method", type=str, default = 'one', help = 'CA Strategy')
    parser.add_argument("--hidden", type=int, default=256, help="Early Stopping Criteria")
    parser.add_argument("--loss", type=str, default = 'bce', help = 'Loss Strategy')
    parser.add_argument("--retrieval", type=str, default = 'ours', help = 'Retrieval')
    parser.add_argument("--K", type=int, default=3, help="Number of Candidates")

    return parser.parse_args()

def training_config(args):
    train_config = dict()
    for arg in vars(args):
        train_config[arg] = getattr(args,arg)
    return train_config

def exp_get_name_RetroPLEX(train_config):
    name = ''
    dic = train_config
    config = ["K","loss","retrieval", "split", "seed","gnn", "lr", "batch_size","embedder"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name
