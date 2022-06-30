import time
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
from torch_sparse import SparseTensor

from utils import time_wrapper, create_slices, sparse_min_max_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_feats: int,
        num_edges: int,
        num_heads: int = 1,
        batch_size: int = 10000,
        row_norm: bool = False,
    ):
        """
        https://stackoverflow.com/questions/20983882/efficient-dot-products-of-large-memory-mapped-arrays
        """
        super(DotProductAttention, self).__init__()
        assert num_heads > 0, 'args.ATTN_HEADS must be >0'

        self.num_nodes = int(num_nodes)
        self.num_feats = int(num_feats)
        self.num_edges = int(num_edges)
        self.num_heads = int(num_heads)
        self.batch_size = int(batch_size)
        self.row_norm = row_norm

        # dot product
        self.out_shape = (self.num_heads, self.num_nodes,
                          self.num_nodes)  # attn shape (w/head)
        self.d_k = self.num_feats * self.num_heads  # hidden dim
        self.scale = 1.0/np.sqrt(self.num_feats)  # scaling factor per head
        self.qk_lin = nn.Linear(self.num_feats, 2*self.d_k)

    def reset_parameters(self):
        self.qk_lin.reset_parameters()

    def _batch_matmul(self, A, B, edge_index):

        # compute dotproduct in batches, across heads
        h_idx = torch.tensor(range(self.num_heads))

        def matmul(n1, n2): return torch.sum(n1*n2, dim=-1)

        values = torch.concat(
            [matmul(
                A[:, edge_index[0, batch], :],
                B[:, edge_index[1, batch], :])
             for batch in create_slices(self.num_edges, self.batch_size)], dim=-1)

        return torch.sparse_coo_tensor(
            indices=torch.stack([
                h_idx.repeat_interleave(self.num_edges),  # h_idx
                edge_index[0].repeat(self.num_heads),  # r_idx
                edge_index[1].repeat(self.num_heads),  # c_idx
            ]),
            values=values.flatten(),
            size=self.out_shape,
        )

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
            h=self.num_heads, hdim=2*self.num_feats
        )

        # separate q and k matrices
        sep_qk = 'L h (split hdim) -> split h L hdim'
        q, k = rearrange(qk, sep_qk, split=2)
        del qk

        # calculate block dot product attention (Q x K^T)/sqrt(dk)
        # k = k.permute([0, 2, 1])  # h L hdim -> h hdim L
        attn = self._batch_matmul(q, k, edge_index)
        del q, k

        # soft max
        attn = torch.sparse.softmax(attn, dim=2)
        attn = (torch.sparse.sum(attn, dim=0)/self.num_heads).coalesce()

        # min-max normalization
        if self.row_norm == True:
            start = time.time()
            attn = sparse_min_max_norm(attn)
            print('Normalization Time: {:0.4f}'.format(time.time()-start))
            return attn

        row, col = attn.indices()

        return SparseTensor(
            row=row,
            col=col,
            value=attn.values().detach(),
            sparse_sizes=list(attn.shape)
        )

@time_wrapper
@torch.no_grad()
def dotproduct_filter(data, args):
    """ create multi-head dot-product attention filter """

    model = DotProductAttention(
        data.num_nodes,
        data.num_features,
        data.num_edges,
        args.ATTN_HEADS,
        batch_size=args.FILTER_BATCH_SIZE,
        row_norm=args.ATTN_NORMALIZATION,
    )

    return model(data.x, data.edge_index)
