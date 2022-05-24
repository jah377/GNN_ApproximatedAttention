import argparse
import numpy as np
from einops import rearrange, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from general.utils import set_seeds, standardize_dataset

parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='products')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--model_name', type=str, default='original')
args = parser.parse_args()


class Original(nn.Module):

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
        self.qkv_lin = nn.Linear(d_m, 3*self.d_k)

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
            hdim=3*self.d_m  # includes q, k, v
        )

        # dot product attention (Q x K^T)/sqrt(dk)
        q, k, _ = rearrange(qkv, 'L h (split hdim) -> split h L hdim', split=3)
        del qkv
        attn = (q @ rearrange(k, 'h L dm -> h dm L')) / self.scale
        attn = F.softmax(attn, dim=-1)

        # mask attn of non-edges and normalize by row
        S = torch.sparse_coo_tensor(
            edge_index,
            torch.ones_like(edge_index[1]),
            size=attn.shape[1:],  # L x L
        ).coalesce()  # sparse mask for single head

        S = torch.stack([S for _ in range(self.num_heads)]
                        ).coalesce()  # sparse mask for all heads

        # mask non-edge attn values per head
        attn = attn.sparse_mask(S).to_dense()

        # normalize by row
        attn = attn.div(reduce(attn, 'h Li Lj -> h Li 1', 'sum'))
        # avg across heads
        attn = reduce(attn, 'h Li Lj -> Li Lj', 'mean')

        attn = attn.to_sparse_coo()  # convert to sparse
        r, c = attn.indices()

        return SparseTensor(
            row=r,
            col=c,
            value=attn.values().detach(),
            sparse_sizes=attn.size()
        )  # to replace adj


class Truncated(nn.Module):

    def __init__(self, d_m: int, num_heads: int = 1):
        """
          d_m:          feature embedding dimension
          num_heads:    attn heads (default=1)
          bias:         learn additive bias (default=True) 
        """
        super().__init__()

        self.d_m = d_m
        self.num_heads = num_heads
        self.proj = d_m//num_heads+1  # min dim permitting num_heads
        self.d_k = self.proj * num_heads
        self.scale = 1.0/np.sqrt(d_m)  # scaling factor per head
        self.qk_lin = nn.Linear(d_m, 2*self.d_k)

    def forward(self, x, edge_index):
        """
          x:          feature embeddings per node [L x dm]
          edge_index: connections [row_idx, col_idx]
        """

        # linear layer + split into heads
        qk = self.qk_lin(x)

        qk = rearrange(
            qk,
            'L (h hdim) -> L h hdim',
            h=self.num_heads,
            hdim=2*self.proj
        )

        # dot product attention (Q x K^T)/sqrt(dk)
        q, k = rearrange(qk, 'L h (split hdim) -> split h L hdim', split=2)
        del qk

        attn = (q.matmul(k.permute([0, 2, 1]))).div(
            self.scale)  # 'h L dm -> h dm L'
        del q, k

        attn = F.softmax(attn, dim=-1)

        # mask attn of non-edges and normalize by row
        S = torch.sparse_coo_tensor(
            edge_index,
            torch.ones_like(edge_index[1]),
            size=attn.shape[1:],  # L x L
        ).coalesce()  # sparse mask for single head

        attn = torch.stack([attn[i].sparse_mask(S)
                           for i in range(self.num_heads)]).to_dense()

        attn = attn.div(attn.sum(dim=-1))  # normalize by row
        attn = attn.mean(dim=0)  # avg heads

        attn = attn.to_sparse_coo()  # convert to sparse
        r, c = attn.indices()

        return SparseTensor(
            row=r,
            col=c,
            value=attn.values().detach(),
            sparse_sizes=attn.size()
        )


def main(args):

    set_seeds(args.seed)
    path = f'data/{args.dataset}_sign_k0.pth'
    data = standardize_dataset(torch.load(path), args.dataset)

    if args.model_name.lower() == 'original':
        NET = Original
    elif args.model_name.lower() == 'truncated':
        NET = Truncated

    model = NET(data.num_features, args.attn_heads)
    attn = model(data.x, data.edge_index)

    return attn

if __name__=='__main__':
    main(args)