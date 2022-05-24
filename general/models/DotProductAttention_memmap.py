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
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.num_edges = num_edges
        self.num_heads = num_heads

        # dot product
        self.out_shape = (num_heads, num_nodes, num_nodes)
        self.proj = int(num_feats//num_heads)+1  # min dim permitting num_heads
        self.d_k = int(self.proj * num_heads)    # hidden dim. of proj. subspace

        self.scale = 1.0/np.sqrt(num_feats)      # scaling factor per head
        self.qk_lin = nn.Linear(num_feats, 2*self.d_k)

    def _batch_slices(self):
        """ idx slices form batch size """
        count = 0
        while True:
            yield slice(count, count + int(self.dpa_batch_size), 1)
            count += int(self.dpa_batch_size)
            if count >= int(self.num_edges):
                break

    def _batch_matmul(self, Q, K, edge_index):
        """ batch matmul from np.memmap """
        # assert isinstance(Q, np.memmap)
        # assert isinstance(K, np.memmap)

        # store attention weights
        attn = torch.sparse_coo_tensor(size=self.out_shape).cpu()

        # compute dotproduct in batches, across heads
        for i in range(self.num_nodes):
            
            q_idx, k_idx = edge_index[:, batch]  # edge_idx -> node_idx
            # q_batch = torch.from_numpy(Q[:, q_idx, :]).to(
            #     device)   # disk -> CPU/GPU
            # k_batch = torch.from_numpy(K[:, :, k_idx]).to(
            #     device)   # disk -> CPU/GPU

            q_batch = Q[:, q_idx, :].to(device)
            k_batch = K[:, :, k_idx].to(device)
            out = (q_batch @ k_batch).cpu().diagonal().T.flatten()

            # store to sparse coo tensor
            r_index = q_idx.repeat(1, self.num_heads).flatten()
            c_index = k_idx.repeat(1, self.num_heads).flatten()
            h_index = torch.tensor(range(self.num_heads)
                                   ).repeat_interleave(q_idx.shape[0])
            assert (r_index.shape == c_index.shape) & (
                r_index.shape == h_index.shape)

            attn += torch.sparse_coo_tensor(
                indices=torch.stack([h_index, r_index, c_index]),
                values=out.flatten(),
                size=self.out_shape,
            )

        return attn  # torch.sparse_coo_tensor

    def forward(self, x, edge_index):
        """
          x:          feature embeddings per node [L x dm]
          edge_index: connections [row_idx, col_idx]
        """
        # linear layer
        qk = self.qk_lin(x)

        # separate heads
        sep_heads = 'L (h hdim) -> L h hdim'
        qk = rearrange(
            qk, sep_heads,
            h=self.num_heads, hdim=2*self.proj)

        # separate q and k
        sep_qk = 'L h (split hdim) -> split h L hdim'
        q, k = rearrange(qk, sep_qk, split=2)
        del qk

        # # move to disk to free up cpu
        # Q = np.memmap(
        #     'q_file.dat', dtype='float16',
        #     mode='w+', shape=q.shape
        #     )
        # K = np.memmap(
        #     'k_file.dat', dtype='float16',
        #     mode='w+', shape=k.shape
        #     )
        # Q[:] = q.detach().numpy() # to disk
        # K[:] = k.permute([0, 2, 1]).detach().numpy()  # to disk
        # del q, k

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
