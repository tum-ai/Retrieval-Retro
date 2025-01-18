from itertools import groupby
import argparse
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", '-d', type=int, default=2, help="GPU to use")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of Epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--eval", type=int, default=5, help="evaluation step")
    parser.add_argument("--es", type=int, default=30, help="Early Stopping Criteria")
    parser.add_argument("--embedder", type=str, default="mpc", help="Early Stopping Criteria")
    parser.add_argument("--input_dim", type=int, default=83, help="Early Stopping Criteria")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Early Stopping Criteria")
    parser.add_argument("--output_dim", type=int, default=256, help="Early Stopping Criteria")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help = 'weight decay')
    parser.add_argument("--seed", type=int, default=0, help = 'Random seed')
    parser.add_argument("--num_sub", type=int, default=2, help = 'Random seed')
    parser.add_argument("--split", type=str, default = 'year', help = 'Dataset Split Strategy')
    parser.add_argument("--lb", type=str, default = 'one', help = 'LB Strategy')
    parser.add_argument("--loss", type=str, default = 'adaptive', help = 'Loss Strategy')
    parser.add_argument("--retrieval", type=str, default = 'ours', help = 'Retrieval')

    return parser.parse_args()


def training_config(args):
    train_config = dict()
    for arg in vars(args):
        train_config[arg] = getattr(args,arg)
    return train_config


def train_mpc(train_config):
    name = ''
    dic = train_config
    config = ["loss","lb", "split", "input_dim", "seed", "lr", "batch_size", "hidden_dim", "embedder"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name
