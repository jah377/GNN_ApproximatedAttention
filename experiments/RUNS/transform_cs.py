
import numpy as np

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from utils import sparse_min_max_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cosine_filter(x, edge_index, args):
    """ create cosine similiarity attention filter """

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    attn_size = (num_nodes, num_nodes)
    cs_scores = np.empty(num_edges)

    for i in range(num_edges):
        A, B = x[edge_index[:, i]].to(device)
        cs_scores[i] = F.cosine_similarity(A, B, dim=-1)

        del A, B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # remove idx of cs_score
    nonzero_idx = np.nonzero(cs_scores != 0)[0]
    attn = torch.sparse_coo_tensor(
        edge_index[:, nonzero_idx],
        torch.from_numpy(cs_scores[nonzero_idx]),
        attn_size)

    # min-max normalization
    if args.ATTN_NORMALIZATION == True:
        return sparse_min_max_norm(attn)

    return SparseTensor.from_torch_sparse_coo_tensor(attn)
