
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from SIGNff_utils import create_slices, sparse_min_max_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def cosine_filter(x, edge_index, args):
#     """ create cosine similiarity attention filter """

#     num_nodes = x.shape[0]
#     num_edges = edge_index.shape[1]
#     attn_size = (num_nodes, num_nodes)
#     cs_scores = np.empty(num_edges)

#     for i in range(num_edges):
#         A, B = x[edge_index[:, i]].to(device)
#         cs_scores[i] = F.cosine_similarity(A, B, dim=-1)

#         del A, B
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # remove idx of cs_score
#     nonzero_idx = np.nonzero(cs_scores != 0)[0]
#     attn = torch.sparse_coo_tensor(
#         edge_index[:, nonzero_idx],
#         torch.from_numpy(cs_scores[nonzero_idx]),
#         attn_size)

#     # min-max normalization
#     if args.ATTN_NORMALIZATION == True:
#         return sparse_min_max_norm(attn)

#     return SparseTensor.from_torch_sparse_coo_tensor(attn)


def cosine_filter(x, edge_index, args):
    """ create cosine similiarity attention filter """

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    attn_size = (num_nodes, num_nodes)

    # pairwise cosine similarity
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    start = time.time()
    cs_scores = torch.concat(
        [cos(x[edge_index[0, batch]], x[edge_index[1, batch]]) for batch in create_slices(num_edges, args.FILTER_BATCH_SIZE)])

    # keep non-zero attention weights
    nonzero_idx = torch.nonzero(cs_scores).flatten()
    attn = torch.sparse_coo_tensor(
        edge_index[:, nonzero_idx],
        cs_scores[nonzero_idx],
        attn_size)

    # min-max normalization
    if args.ATTN_NORMALIZATION == True:
        start = time.time()
        attn = sparse_min_max_norm(attn)
        print('Normalization Time: {:0.4f}'.format(time.time()-start))
        return attn

    return SparseTensor.from_torch_sparse_coo_tensor(attn)
