from ..config import Config
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import random
import torch.nn.functional as F
import math



def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b,l, h, w = matrices.size()

    indices = torch.triu_indices(l, w, offset=0 if mask_diagonal else 1)
    matrices[:,indices[0],:, indices[1]] = maskval

class Multi_Head_Attention(nn.Module):
    def __init__(self, config):
        super(Multi_Head_Attention, self).__init__()
        self.model_dim = config.model_dim
        self.head_dim = (self.model_dim // config.num_head)
        self.num_head = config.num_head
        assert self.head_dim*self.num_head == self.model_dim, 'Improper number of heads'

        self.to_query = nn.Linear(self.model_dim, self.head_dim*self.num_head)
        self.to_key = nn.Linear(self.model_dim, self.head_dim*self.num_head)
        self.to_value = nn.Linear(self.model_dim, self.head_dim*self.num_head)

        self.to_out = nn.Linear(self.head_dim*self.num_head, self.model_dim)

    def forward(self, Q, K, V, mask = False):
        # Q: (N, qlen, dm) K: (N, klen, dm) V:(N, vlen, dm)
        N, qlen, dm = Q.size()
        _, klen, _ = K.size()
        _, vlen, _ = V.size()
        assert dm == self.model_dim, 'Improper size model dimmention'
        # apply linear projection
        q = self.to_query(Q).view(N, qlen, self.num_head, -1) # q: (N, qlen, h, dh)
        k = self.to_query(K).view(N, klen, self.num_head, -1) # k: (N, klen, h, dh)
        v = self.to_query(Q).view(N, vlen, self.num_head, -1) # v: (N, vlen, h, dh)
        dot = torch.einsum("bqhd,bkhd->bqhk", [q,k]).contiguous()/math.sqrt(head_dim) # dot: (N, qlen, h, klen)
        if mask:
            mask_(dot, float('-inf'), mask_diagonal=False)
        weights = F.softmax(dot.float(), dim = -1).type_as(dot) # weights: (N, qlen, h, klen)
        out = torch.einsum("bqhk,bkhd->bqhd", [weights, v]).contiguous() # out: (N, qlen, h, dh)
        out = out.view(N, qlen, -1) # out: (N, qlen, h*dh)
        out = self.to_out(out) # out: (N, qlen, dm)
        return out
