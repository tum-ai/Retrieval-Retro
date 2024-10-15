from cmath import nan
import numpy as np
import argparse
from scipy import stats
import torch
from sklearn.metrics import r2_score
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#R square 
def r2(x1, x2):
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()
    return r2_score(x1.flatten(), x2.flatten(), multioutput='variance_weighted')

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)

    return (data - data_min) / (data_max - data_min)

def test(emb_net, data_loader):
    emb_net.eval()
    list_embs = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.cuda()
            _, embs = emb_net(batch)
            embs = F.normalize(embs, p=2, dim=1)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)

def get_numerical_data_loader(*data, batch_size, shuffle=False, dtype=torch.float32):
    tensors = [torch.tensor(d, dtype=dtype) for d in data]
    return DataLoader(TensorDataset(*tuple(tensors)), batch_size=batch_size, shuffle=shuffle)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", '-d', type=int, default=6, help="GPU to use")
    parser.add_argument("--seed", type=int, default=0, help = 'Random seed')
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of Epochs for training")
    parser.add_argument("--emb_epochs", type=int, default=500, help="Number of Epochs for emb net training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size") # (16 --> 2000) (64 --> 6000)
    parser.add_argument("--layers", "-l", type=int, default=3, help="The number layers of the Processor")
    parser.add_argument("--eval", type=int, default=5, help="evaluation step")
    parser.add_argument("--es", type=int, default=50, help="Early Stopping Criteria")
    parser.add_argument("--embedder", type=str, default="graphnetwork", help="GNNs")
    parser.add_argument("--hidden", type=int, default=256, help="Early Stopping Criteria")
    parser.add_argument("--pretrain", type=str, default='formation', help="pretraining_task")
    return parser.parse_args()


def training_config(args):
    train_config = dict()
    for arg in vars(args):
        train_config[arg] = getattr(args,arg)
    return train_config


def train_predictor(train_config):
    name = ''
    dic = train_config
    config = ["pretrain","embedder", "lr", "batch_size", "hidden", "seed"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name