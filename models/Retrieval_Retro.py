import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from torch_geometric.nn import Set2Set
from models import GraphNetwork
from torch_scatter import scatter_sum
from collections import Counter
import copy
from .layers import Self_TransformerEncoder_non, Cross_TransformerEncoder_non
import numpy as np


class Retrieval_Retro(nn.Module):
    def __init__(self, gnn, layers, input_dim, output_dim, hidden_dim, n_bond_feat, device, t_layers, t_layers_sa, num_heads):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.classifier = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim*3), nn.PReLU(), nn.Linear(hidden_dim*3, self.output_dim), nn.Sigmoid())
        self.gnn = GraphNetwork(layers, input_dim, n_bond_feat, hidden_dim, device)

        # MPC
        self.self_attention = Self_TransformerEncoder_non(hidden_dim, num_heads, t_layers_sa, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.cross_attention = Cross_TransformerEncoder_non(hidden_dim, num_heads, t_layers, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.fusion_linear = nn.Sequential(nn.Linear(512, 256), nn.PReLU())

        # NRE
        self.self_attention_2 = Self_TransformerEncoder_non(hidden_dim, num_heads, t_layers_sa, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.cross_attention_2 = Cross_TransformerEncoder_non(hidden_dim, num_heads, t_layers, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.fusion_linear_2 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())

        self.init_model()


    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):


        main_graph = data[0].to(self.device)
        additional_graph = data[1]
        additional_graph_2 = data[2]


        main_graph_x = self.gnn(main_graph)
        main_weighted_x = main_graph_x * main_graph.fc_weight.reshape(-1, 1)
        main_graph_emb = scatter_sum(main_weighted_x, main_graph.batch, dim = 0)

        # For additional_graph 
        add_graph_outputs = []

        for graph in additional_graph:
            add_graph = graph.to(self.device)

            add_graph_x = self.gnn(add_graph)
            add_weighted_x = add_graph_x * add_graph.fc_weight.reshape(-1, 1)
            add_graph_emb = scatter_sum(add_weighted_x, add_graph.batch, dim = 0)
            add_graph_outputs.append(add_graph_emb.unsqueeze(1))

        
        add_pooled = torch.stack(add_graph_outputs, dim=1).squeeze(2)

        add_graph_outputs_2 = []
        for graph in additional_graph_2:
            add_graph = graph.to(self.device)

            add_graph_x = self.gnn(add_graph)
            add_weighted_x = add_graph_x * add_graph.fc_weight.reshape(-1, 1)
            add_graph_emb = scatter_sum(add_weighted_x, add_graph.batch, dim = 0)
            add_graph_outputs_2.append(add_graph_emb.unsqueeze(1))

        
        add_pooled_2 = torch.stack(add_graph_outputs_2, dim=1).squeeze(2)

        # MPC
        # Self Attention Layers
        main_graph_repeat = main_graph_emb.unsqueeze(1).repeat(1,add_pooled.shape[1],1)
        add_pooled = torch.cat([add_pooled, main_graph_repeat], dim=2)
        add_pooled = self.fusion_linear(add_pooled)

        add_pooled_self = self.self_attention(add_pooled)

        # Cross Attention Layers
        cross_attn_output = self.cross_attention(main_graph_emb.unsqueeze(0), add_pooled_self, add_pooled_self)

        # NRE
        # Self Attention Layers
        add_pooled_2 = torch.cat([add_pooled_2, main_graph_repeat], dim=2)
        add_pooled_2 = self.fusion_linear_2(add_pooled_2)

        add_pooled_self_2 = self.self_attention_2(add_pooled_2)

        # Cross Attention Layers
        cross_attn_output_2 = self.cross_attention_2(main_graph_emb.unsqueeze(0), add_pooled_self_2, add_pooled_self_2)


        classifier_input = torch.cat([main_graph_emb, cross_attn_output.squeeze(0), cross_attn_output_2.squeeze(0)], dim=1)
        template_output = self.classifier(classifier_input)

        return template_output    
