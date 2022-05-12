import numpy as np
from einops import rearrange, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor


class net(nn.Module):

    def __init__(self, d_m: int, num_heads: int = 1):
        """
          d_m:          feature embedding dimension
          num_heads:    attn heads (default=1)
          bias:         learn additive bias (default=True) 
        """
        super().__init__()

        self.d_m = d_m
        self.num_heads = num_heads
        self.d_k = num_heads * d_m      # hidden dim. of proj. subspace
        self.scale = 1.0/np.sqrt(d_m)   # scaling factor per head
        self.qkv_lin = nn.Linear(d_m, 3*self.d_k) # stacked q,k,v for efficiency

    def forward(self, x, edge_index):
        """
          x:          feature embeddings per node [L x dm]
          edge_index: connections [row_idx, col_idx]
        """

        # linear layer + split into heads
        qkv = self.qkv_lin(x)

        qkv = rearrange(
            qkv,
            'L (h hdim) -> L h hdim', 
            h=self.num_heads,
            hdim=3*self.d_m # includes q, k, v
        )

        # dot product attention (Q x K^T)/sqrt(dk)
        q, k, _ = rearrange(qkv, 'L h (split hdim) -> split h L hdim', split=3)
        attn = (q @ rearrange(k, 'h L dm -> h dm L')) / self.scale
        attn = F.softmax(attn, dim=-1)

        # mask attn of non-edges and normalize by row
        S = torch.sparse_coo_tensor(
              edge_index, 
              torch.ones_like(edge_index[1]),
              size=attn.shape[1:], # L x L
              ).coalesce() # sparse mask for single head 

        S = torch.stack([S for _ in range(self.num_heads)]).coalesce() # sparse mask for all heads

        attn = attn.sparse_mask(S).to_dense() # mask non-edge attn values per head

        attn = attn.div( reduce(attn, 'h Li Lj -> h Li 1', 'sum') ) # normalize by row
        attn = reduce(attn, 'h Li Lj -> Li Lj', 'mean')             # avg across heads
        attn = attn.to_sparse_coo()                                 # convert to sparse

        return SparseTensor(
            row=attn.indices()[0], 
            col=attn.indices()[1], 
            value=attn.values().detach(),
            sparse_sizes=attn.size()
            ) # to replace adj
