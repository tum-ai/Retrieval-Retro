import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum
from .layers import TransformerEncoder

class MPC(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, device):
        super(MPC, self).__init__()
        
        self.comp_encoder = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden), nn.PReLU(), nn.Linear(n_hidden, 32))
        self.partial_comp_encoder = nn.Sequential(nn.Linear(output_dim, 32))

        self.output_dim = output_dim
        self.precursor_matrix = nn.Embedding(output_dim, 32)

        # For attention block
        self.transformerblock = TransformerEncoder(embed_dim=32, num_heads=8, layers=2, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        # For reconstruction
        self.decoder = nn.Sequential(nn.Linear(32, n_hidden),nn.Linear(n_hidden, n_hidden), nn.Linear(n_hidden, input_dim))
        self.sigmoid = nn.Sigmoid()
        self.device = device

        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, kb=None):
        
        main_graph = data.to(self.device)
        composition_vector = main_graph.comp_fea.reshape(len(main_graph.ptr)-1, -1)
        encoding = self.comp_encoder(composition_vector)
        recon_output = self.decoder(encoding)

        precursor_num = torch.tensor(np.arange(self.output_dim)).to(self.device)
        precursor_matrix = self.precursor_matrix(precursor_num).repeat(encoding.shape[0], 1, 1)

        masked_tensor = torch.zeros_like(precursor_matrix).to(self.device)

        if self.training:

            precursor_vector = main_graph.y_lb_one.reshape(len(main_graph.ptr)-1, -1)
            precursor_idx = precursor_vector.nonzero() # Most frequent one
            masked_precursor_idx = remove_random_index_per_group(precursor_idx).cuda()
            zero_tensor = torch.zeros_like(precursor_vector).cuda()
            for idx in masked_precursor_idx:
                row_idx = idx[0]
                col_idx = idx[1]

                zero_tensor[row_idx][col_idx] = precursor_vector[row_idx][col_idx]
            partial_precursor_vector = zero_tensor       

            # Masking precursor matrix
            for idx in partial_precursor_vector.nonzero():
                masked_tensor[idx[0], idx[1], :] = precursor_matrix[idx[0], idx[1], :]

            encoding_attn = self.transformerblock(encoding.unsqueeze(0), masked_tensor.permute(1,0,2), masked_tensor.permute(1,0,2)).permute(1,0,2)

            dot_product = torch.bmm(encoding_attn, precursor_matrix.permute(0,2,1)).squeeze(1)
            probability = torch.sigmoid(dot_product)

        else:
            if kb==None:
                encoding_attn = self.transformerblock(encoding.unsqueeze(0), precursor_matrix.permute(1,0,2), precursor_matrix.permute(1,0,2)).permute(1,0,2)

                dot_product = torch.bmm(encoding_attn, precursor_matrix.permute(0,2,1)).squeeze(1)
                probability = torch.sigmoid(dot_product)

        return probability,encoding, recon_output



def remove_random_index_per_group(tensor):

    unique_indices = torch.unique(tensor[:, 0])

    keep_indices = []

    for index in unique_indices:
        occurrences = (tensor[:, 0] == index).nonzero(as_tuple=True)[0]
        to_remove = torch.randperm(occurrences.size(0))[0]

        keep_indices.extend([i.item() for i in occurrences if i != occurrences[to_remove]])

    new_tensor = tensor[keep_indices]

    return new_tensor


class CircleLoss(nn.Module):
    def __init__(self, gamma=64, margin=0.25):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

        self.O_p = 1 + margin
        self.O_n = -margin
        self.Delta_p = 1 - margin
        self.Delta_n = margin


    def forward(self, y_true, y_pred):

        alpha_p = F.relu(self.O_p - y_pred.detach())
        alpha_n = F.relu(y_pred.detach() - self.O_n)

        y_true = y_true.float()

        logit_p = -y_true * (alpha_p * (y_pred - self.Delta_p)) * self.gamma
        logit_p = logit_p - (1.0 - y_true) * 10000.0
        loss_p = torch.logsumexp(logit_p, dim=-1)

        logit_n = (1.0 - y_true) * (alpha_n * (y_pred - self.Delta_n)) * self.gamma
        logit_n = logit_n - y_true * 10000.0
        loss_n = torch.logsumexp(logit_n, dim=-1)

        loss = F.softplus(loss_p + loss_n)

        return loss.mean()
    
    
class MultiLossLayer(nn.Module):
    def __init__(self, task_names, device):
        super(MultiLossLayer, self).__init__()
        self.task_names = task_names
        self.num_task = len(self.task_names)
        self.device = device
        # Initialize log_vars
        initial_log_vars = np.zeros((self.num_task,), dtype=np.float32)
        self.log_vars = nn.Parameter(torch.tensor(initial_log_vars)).to(device)

    def forward(self, inputs):
        precision = torch.exp(-self.log_vars)
        multi_loss = torch.sum(inputs * precision, dim=-1) + torch.sum(self.log_vars, dim=-1)
        return multi_loss
