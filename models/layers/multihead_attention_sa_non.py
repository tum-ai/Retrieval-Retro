import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
# Code adapted from the fairseq repo.

class MultiheadAttention_sa_non(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.scaling = self.embed_dim ** -0.5
        
        # self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        # self.register_parameter('in_proj_bias', None)
        # if bias:
        #     self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # if add_bias_kv:
        #     self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
        #     self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        # else:
            # self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # self.reset_parameters() # precursor encoding때 안했었음

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        # query = query.unsqueeze(0)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()


        # if qkv_same:
        #     q, k, v = self.in_proj_qkv(query)
        # # else:
        # #     raise NotImplementedError
            
        # else:
        #     q = self.in_proj_q(query)
        #     k = self.in_proj_k(key)
        #     v = self.in_proj_v(value)
        # q = q * self.scaling
        

        # q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # print('multi head attention')
        # Multi head attention
        attn_weights = torch.bmm(query, key.transpose(1,2)) * self.scaling
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, value)
        # attn = attn.transpose(0, 1)

        return attn


    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)