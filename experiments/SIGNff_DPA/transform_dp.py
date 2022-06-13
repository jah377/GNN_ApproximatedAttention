import numpy as np
from scipy import sparse
from einops import rearrange

import torch
import torch.nn as nn
from torch_sparse import SparseTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_feats: int,
        num_edges: int,
        num_heads: int = 1,
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
        values = torch.zeros(self.num_edges*self.num_heads)

        start, end = 0, self.num_heads
        for i in range(self.num_edges):
            r_idx, c_idx = edge_index[:, i]
            A_node = A[:, r_idx, :].unsqueeze(dim=1).to(device)  # to gpu
            B_node = B[:, :, c_idx].unsqueeze(dim=2).to(device)  # to gpu

            values[start:end] = A_node.matmul(
                B_node).detach().flatten().cpu()
            start += self.num_heads
            end += self.num_heads

        return torch.sparse_coo_tensor(
            indices=torch.stack([
                h_idx.repeat(self.num_edges),  # h_idx
                edge_index[0].repeat_interleave(self.num_heads),  # r_idx
                edge_index[1].repeat_interleave(self.num_heads),  # c_idx
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
        k = k.permute([0, 2, 1])  # h L hdim -> h hdim L
        attn = self._batch_matmul(q, k, edge_index)
        del q, k

        # soft max
        attn = torch.sparse.softmax(attn, dim=2)
        attn = (torch.sparse.sum(attn, dim=0)/self.num_heads).coalesce()

        # min-max normalization
        if self.row_norm == False:
            row, col = attn.indices()

            return SparseTensor(
                row=row,
                col=col,
                value=attn.values().detach(),
                sparse_sizes=list(attn.shape)
            )

        return sparse_min_max_norm(attn)


def sparse_min_max_norm(s_coo):
    """ row-level min-max normalization """
    # https://stackoverflow.com/questions/51570512/minmax-scale-sparse-matrix-excluding-zero-elements
    assert type(s_coo) == torch.Tensor, f'{type(s_coo)} not torch.Tensor'
    assert s_coo.layout == torch.sparse_coo, f'{s_coo.type} not in torch.sparse_coo format'

    # convert to scipy.sparse.csr_matrix
    if not s_coo.is_coalesced():
        s_coo = s_coo.coalesce()
    r, c = s_coo.indices()
    v = s_coo.values()
    shape = list(s_coo.shape)
    s_csr = sparse.csr_matrix((v, (r, c)), shape=shape)

    # calculate min-max per row
    v = s_csr.data
    nnz = s_csr.getnnz(axis=1)          # total edges per row
    idx = np.r_[0, nnz[:-1].cumsum()]   # v.idx corresponding to each row

    max_r = np.maximum.reduceat(v, idx) # max per row
    min_r = np.minimum.reduceat(v, idx) # min per row
    min_r *= (nnz == shape[1])          # if not fully-connected, min=0

    # create matrices for vectorization
    max_m = np.repeat(max_r, nnz)
    min_m = np.repeat(min_r, nnz)

    return SparseTensor(
        row=r,
        col=c,
        value=torch.tensor((v-min_m)/(max_m-min_m)),
        sparse_sizes=shape
    )


def dotproduct_filter(data, args):
    """ create multi-head dot-product attention filter """
    assert hasattr(args, 'ATTN_HEADS'), 'Must specify ATTN_HEADS value'
    assert hasattr(
        args, 'DPA_NORMALIZATION'), 'Must specify DPA_NORMALIZATION value'

    model = DotProductAttention(
        data.num_nodes,
        data.num_features,
        data.num_edges,
        args.ATTN_HEADS,
        args.DPA_NORMALIZATION,
    )

    return model(data.x, data.edge_index)
