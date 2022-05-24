"""
Block Dot: https://gist.github.com/kostrykin/0812813bb2d48f9e419f3ae1f60ba43f

Stack: https://stackoverflow.com/questions/20983882/efficient-dot-products-of-large-memory-mapped-arrays/21096605#21096605

EXPLORE: https://github.com/nikulukani/pycublasxt
"""
# %%
import torch.nn.functional as F
from general.utils import set_seeds, standardize_dataset
import time
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from general.models.DotProductAttention_memmap import net
import os
# %%
os.chdir('/home/jharris/Desktop/approx_attention/')

# %% PREPARE CORA DATA


class Args(Dataset):
    def __init__(self, seed, dataset):
        self.seed = seed
        self.dataset = dataset.lower()


args = Args(
    seed=42,
    dataset='cora'
)

# download data
set_seeds(args.seed)
path = f'data/{args.dataset}/{args.dataset}_sign_k0.pth'
data = standardize_dataset(torch.load(path), args.dataset)


model = net(data.num_nodes, data.num_features, num_heads=4)
model(data.x, data.edge_index)


# compute Q and K
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
k = k.permute([0, 2, 1])  # rearrrange for matrix multiplication
q, k = q.type(torch.float16), k.type(torch.float16)
# torch_matmul = (q @ k)

# convert to memmap
qpath = os.path.join(os.getcwd(), 'BLOCK', 'q_file.dat')
kpath = os.path.join(os.getcwd(), 'BLOCK', 'k_file.dat')
spath = os.path.join(os.getcwd(), 'BLOCK', 'store_file.dat')

A = np.memmap(qpath, dtype='float16', mode='w+', shape=q.shape)
B = np.memmap(kpath, dtype='float16', mode='w+', shape=k.shape)
A[:] = q.detach().numpy()
B[:] = k.detach().numpy()

# # %% TESTING

# def _block_slices(dim_size, block_size):
#     count = 0
#     while True:
#         yield slice(count, count + int(block_size), 1)
#         count += int(block_size)
#         if count >= int(dim_size):
#             break


# def blockwise_dot(A, B, max_elements=int(2**27), out=None):
#     """
#     Computes the dot product of two matrices in a block-wise fashion.
#     Only blocks of `A` with a maximum size of `max_elements` will be
#     processed simultaneously.
#     """

#     h, m,  n = A.shape
#     _, n1, o = B.shape

#     if n1 != n:
#         raise ValueError('matrices are not aligned')

#     if A.flags.f_contiguous:
#         # prioritize processing as many columns of A as possible
#         max_cols = max(1, max_elements / m)
#         max_rows = max_elements / max_cols
#     else:
#         # prioritize processing as many rows of A as possible
#         max_rows = max(1, max_elements / n)
#         max_cols = max_elements / max_rows

#     if out is None:
#         out = np.empty((h, m, o), dtype=np.result_type(A, B))
#     elif out.shape != (h, m, o):
#         raise ValueError('output array has incorrect dimensions')

#     for mm in _block_slices(m, max_rows):
#         out[mm, :] = 0
#         for nn in _block_slices(n, max_cols):
#             A_block = A[:, mm, nn].copy()  # copy to force a read
#             out[:, mm, :] = np.matmul(A_block, B[:, nn, :])
#             del A_block

#     if out is None:
#         return out


# max_elements = 2 ** 27

# h, n, f = A.shape
# attn_path = os.path.join(os.getcwd(), 'BLOCK', 'attn_file.dat')
# attn = np.memmap(attn_path, dtype='float16', mode='w+', shape=(h,n,n))
# blockwise_dot(A, B, max_elements=max_elements, out=attn)

# # %%
# import torch.nn.functional as F
# attn = torch.from_numpy(attn/5).type(torch.DoubleTensor)  # to torch
# attn = F.softmax(attn, dim=-1)

# S = torch.sparse_coo_tensor(
#     data.edge_index,
#     torch.ones_like(data.edge_index[1]),
#     size=attn.shape[1:],  # L x L
# ).coalesce()  # sparse mask for single head

# attn = torch.stack([attn[i].sparse_mask(S)
#                     for i in range(num_heads)])
# # %% convert to scipy sparse csr matrix
# """
# https://stackoverflow.com/questions/49286903/normalize-sparse-row-probability-matrix
# """
# attn.sum(dim=-1)


# %%

num_heads = 4
num_nodes = data.num_nodes
num_feats = data.num_features
num_heads = num_heads
max_elements = 2**27

out_shape = (num_heads, num_nodes, num_nodes)
# min dim permitting num_heads
proj = int(num_feats//num_heads)+1
# hidden dim. of proj. subspace
d_k = int(proj * num_heads)

scale = 1.0/np.sqrt(num_feats)   # scaling factor per head
qk_lin = nn.Linear(num_feats, 2*d_k)


# linear layer + split into heads
qk = qk_lin(data.x)

qk = rearrange(
    qk,
    'L (h hdim) -> L h hdim',
    h=num_heads,
    hdim=2*proj
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
attn = np.memmap(
    'attn_file.dat', dtype='float16',
    mode='w+', shape=out_shape
)

A = Q
B = K


if A.flags.f_contiguous:
    # prioritize processing as many columns of A as possible
    max_cols = max(1, max_elements / num_nodes)
    max_rows = max_elements / max_cols
    print(max_rows, max_cols)

else:
    # prioritize processing as many rows of A as possible
    max_rows = max(1, int(max_elements/num_feats))
    max_cols = int(max_elements/max_rows)


def _block_slices(dim_size, block_size):
    """Generator that yields slice objects for indexing into 
    sequential blocks of an array along a particular axis
    """
    count = 0
    while True:
        yield slice(count, count + int(block_size), 1)
        count += int(block_size)
        if count >= int(dim_size):
            break


for r_slice in _block_slices(num_nodes, max_rows):
    attn[:, r_slice, :] = 0
    for c_slice in _block_slices(num_feats, max_cols):
        A_block = A[:, r_slice, c_slice].copy()  # copy to force a read
        attn[:, r_slice, :] += np.matmul(A_block, B[:, c_slice, :])
        del A_block

# %%

attn = torch.from_numpy(attn/scale)  # memmap -> torch
attn = F.softmax(attn.type(torch.DoubleTensor), dim=-1)

# mask attn of non-edges and normalize by row
S = torch.sparse_coo_tensor(
    data.edge_index,
    torch.ones_like(data.edge_index[1]),
    size=attn.shape[1:],  # L x L
).coalesce()  # sparse mask for single head

attn = torch.stack([attn[i].sparse_mask(S)
                    for i in range(num_heads)]).to_dense()
del S

attn = attn.div(attn.sum(dim=-1, keepdim=True))  # normalize by row
attn = attn.mean(dim=0)  # avg heads

attn = attn.to_sparse_coo()  # convert to sparse
r, c = attn.indices()


# %%
