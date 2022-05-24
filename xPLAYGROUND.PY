# %%
from scipy.sparse import csr_matrix
import os
import numpy as np
from einops import rearrange, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch.utils.data import Dataset

from general.utils import set_seeds, standardize_dataset


class Args(Dataset):
    def __init__(self, seed, dataset):
        self.seed = seed
        self.dataset = dataset.lower()


args = Args(
    seed=42,
    dataset='cora'
)

set_seeds(args.seed)
path = f'data/{args.dataset}/{args.dataset}_sign_k0.pth'
data = standardize_dataset(torch.load(path), args.dataset)

num_heads = 4

# %% MHA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MHA(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_edges: int,
            num_feats: int,
            num_heads: int = 1,
            batch_size: int = 1,
    ):
        """
        https://stackoverflow.com/questions/20983882/efficient-dot-products-of-large-memory-mapped-arrays

          num_nodes:    total number of nodes 
          num_feats:    feature embedding dimension
          num_heads:    attn heads (default=1)
          num_edges:    total number of edges  
          batch_size:   n nodes in batch 

        """
        super().__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_feats = num_feats
        self.num_heads = num_heads
        self.batch_size = batch_size

        self.out_shape = (num_heads, num_nodes, num_nodes)
        # min dim permitting num_heads
        self.proj = int(num_feats//num_heads)+1
        # hidden dim. of proj. subspace
        self.d_k = int(self.proj * num_heads)

        self.scale = 1.0/np.sqrt(num_feats)   # scaling factor per head
        self.qk_lin = nn.Linear(num_feats, 2*self.d_k)

    def _batch_slices(self):
        """Generator that yields slice objects for indexing into 
        sequential blocks of an array along a particular axis
        """
        count = 0
        while True:
            yield slice(count, count + int(self.batch_size), 1)
            count += int(self.batch_size)
            if count >= int(self.num_edges):
                break

    def _batch_matmul(self, Q, K, edge_index):
        """
        Computes the matrix multiplication of two matrices in a block-wise fashion. 
        Only blocks of `A` with a maximum size of `max_elements` will be 
        processed simultaneously.

        Args:
            A:              np.memmap
            B:              np.memmap
            edge_index:     torch.tensor
        """
        assert isinstance(Q, np.memmap)
        assert isinstance(K, np.memmap)

        attn = torch.sparse_coo_tensor(size=self.out_shape).cpu()

        for batch in self._batch_slices():
            q_idx, k_idx = edge_index[batch]  # edge_idx -> node_idx
            q_batch = torch.from_numpy(Q[:, q_idx, :]).to(device)
            k_batch = torch.from_numpy(K[:, :, k_idx]).to(device)
            out = torch.matmul(q_batch, k_batch).cpu().item()
            del q_batch, k_batch

            # store to sparse coo tensor
            h_index = torch.tensor(range(self.num_heads)
                                   ).repeat_interleave(self.num_heads)
            r_index, c_index = edge_index.repeat(1, self.num_heads)
            attn += torch.sparse_coo_tensor(
                indices=[h_index, r_index, c_index],
                values=out.flatten(),
                size=self.out_shape,
            )
            del h_index, r_index, c_index, out

        return attn

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

        q, k = rearrange(qk, 'L h (split hdim) -> split h L hdim', split=2)
        del qk

        # calculate block dot product attention (Q x K^T)/sqrt(dk)
        k = k.permute([0, 2, 1])  # h L hdim -> h hdim L

        # convert to memmap
        Q = np.memmap(
            'q_file.dat', dtype='float16',
            mode='w+', shape=q.shape
        )
        K = np.memmap(
            'k_file.dat', dtype='float16',
            mode='w+', shape=k.shape
        )
        Q[:] = q.detach().numpy()
        K[:] = k.detach().numpy()
        del q, k

        # calculate attention
        attn = self._batch_matmul(Q, K, edge_index)
        del Q, K

        return attn

        # self._blockwise_matmul(Q, K, attn)  # write to disk
        # del Q, K

        # attn = torch.from_numpy(attn/self.scale)  # memmap -> torch
        # attn = F.softmax(attn.type(torch.DoubleTensor), dim=-1)

        # # mask attn of non-edges and normalize by row
        # S = torch.sparse_coo_tensor(
        #     edge_index,
        #     torch.ones_like(edge_index[1]),
        #     size=attn.shape[1:],  # L x L
        # ).coalesce()  # sparse mask for single head

        # attn = torch.stack([attn[i].sparse_mask(S)
        #                    for i in range(self.num_heads)]).to_dense()
        # del S
        # attn = attn.div(attn.sum(dim=-1, keepdim=True))  # normalize by row
        # attn = attn.mean(dim=0)  # avg heads
        # attn = attn.to_sparse_coo()  # convert to sparse
        # r, c = attn.indices()
        # return SparseTensor(
        #     row=r,
        #     col=c,
        #     value=attn.values().detach(),
        #     sparse_sizes=attn.size()
        # )


n, f = data.x.shape
e = data.edge_index.shape[1]
model = MHA(n, e, f, num_heads, batch_size=1000)
attn = model(data.x, data.edge_index)  # attn


# %%


# %%
scale = 5
attn = torch.from_numpy(attn)/scale
attn = F.softmax(attn.type(torch.DoubleTensor), dim=-1)


# %% Calculate Q AND K

d_m = data.num_features
num_heads = 4
proj = d_m//num_heads+1  # min dim permitting num_heads
d_k = proj * num_heads
scale = 1.0/np.sqrt(d_m)  # scaling factor per head
qk_lin = nn.Linear(d_m, 2*d_k)

qk = qk_lin(data.x).type(torch.float16)  # reduce memory
qk = rearrange(
    qk,
    'L (h hdim) -> L h hdim',
    h=num_heads,
    hdim=2*proj
)

q, k = rearrange(qk, 'L h (split hdim) -> split h L hdim', split=2)
k = k.permute([0, 2, 1])

# %% numpy


def batch_slices(dim_size, batch_size):
    """Generator that yields slice objects for indexing into 
    sequential blocks of an array along a particular axis
    """
    count = 0
    while True:
        yield slice(count, count + int(batch_size), 1)
        count += int(batch_size)
        if count >= int(dim_size):
            break


num_heads, num_nodes, num_feats = q.shape
out = np.zeros_like(data.edge_index[1])
num_edges = data.edge_index.shape[1]
batch_size = 10

batches = batch_slices(num_edges, batch_size)
batch = next(iter(batches))
A = q[:, batch, :]
B = k[:, :, batch]
out = torch.matmul(A, B).detach()


# %%

row, col = data.edge_index[:, batch]

csr = csr_matrix((out, (row, col)), shape=(num_heads, num_nodes, num_nodes))

# %%
