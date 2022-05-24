import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class net(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_feats: int,
            num_edges: int,
            num_heads: int = 1,
    ):
        """
        https://stackoverflow.com/questions/20983882/efficient-dot-products-of-large-memory-mapped-arrays

          num_nodes:    total number of nodes 
          num_feats:    feature embedding dimension
          num_heads:    attn heads (default=1)

        """
        super().__init__()
        assert num_heads > 0

        # definitions
        self.num_nodes = int(num_nodes)
        self.num_feats = int(num_feats)
        self.num_edges = int(num_edges)
        self.num_heads = int(num_heads)

        # dot product
        self.out_shape = (self.num_heads, self.num_nodes,
                          self.num_nodes)  # attn shape (w/head)
        self.d_k = self.num_feats * self.num_heads  # hidden dim
        self.scale = 1.0/np.sqrt(self.num_feats)  # scaling factor per head
        self.qk_lin = nn.Linear(self.num_feats, 2*self.d_k)

    def _batch_matmul(self, A, B, edge_index):
        # store attention weights
        attn = torch.sparse_coo_tensor(size=self.out_shape).cpu()

        # compute dotproduct in batches, across heads
        for i in range(self.num_nodes):
            h_idx = torch.tensor(range(self.num_heads))
            r_idx, c_idx = edge_index[i]

            A_node = A[:, r_idx, :].to(device)  # to gpu
            B_node = B[:, :, c_idx].to(device)  # to gpu

            attn += torch.sparse_coo_tensor(
                indices=torch.stack([
                    h_idx,
                    r_idx.repeat_interleave(self.num_heads),
                    c_idx.repeat_interleave(self.num_heads)
                ]),
                values=(A_node @ B_node).flatten().cpu(),
                size=self.out_shape,
            )

            del A_node, B_node
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return attn  # torch.sparse_coo_tensor

    def forward(self, x, edge_index):
        """
          x:          feature embeddings per node [L x dm]
          edge_index: connections [row_idx, col_idx]
        """
        # compute linear layer
        qk = self.qk_lin(x)

        # separate attention heads
        sep_heads = 'L (h hdim) -> L h hdim'
        qk = rearrange(
            qk, sep_heads,
            h=self.num_heads, hdim=2*self.proj
        )

        # separate q and k matrices
        sep_qk = 'L h (split hdim) -> split h L hdim'
        q, k = rearrange(qk, sep_qk, split=2)
        del qk

        # calculate block dot product attention (Q x K^T)/sqrt(dk)
        k = k.permute([0, 2, 1])  # h L hdim -> h hdim L
        attn = self._batch_matmul(q, k, edge_index)
        del q, k

        # soft max
        attn = torch.sparse.softmax(attn, dim=2)        # sum(row)=1
        attn = torch.sparse.sum(attn, dim=0)/self.num_heads  # avg across heads

        return SparseTensor(
            row=attn.indices()[0],
            col=attn.indices()[1],
            value=attn.values().detach(),
            sparse_sizes=attn.size()
        )