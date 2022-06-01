# %%
import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F

from general.utils import time_wrapper  # wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = 'data/cora/cora_sign_k0.pth'
data = torch.load(path)



# %%

# calculate adj matrix
row, col = data.edge_index
adj_t = SparseTensor(
    row=col,
    col=row,
    sparse_sizes=(data.num_nodes, data.num_nodes)
)

# setup degree normalization tensors
deg = adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0


