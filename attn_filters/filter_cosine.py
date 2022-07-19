import time

import torch
from torch_sparse import SparseTensor

from utils import time_wrapper, create_slices, sparse_min_max_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@time_wrapper
@torch.no_grad()
def cosine_filter(x, edge_index, args):
    """ create cosine similiarity attention filter """

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    attn_size = (num_nodes, num_nodes)

    # pairwise cosine similarity
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cs_scores = torch.concat([
        cos(x[edge_index[0, batch]], x[edge_index[1, batch]])
        for batch in create_slices(num_edges, args.FILTER_BATCH_SIZE)
    ]).flatten()

    # keep non-zero attention weights
    nonzero_idx = torch.nonzero(cs_scores).flatten()
    attn = torch.sparse_coo_tensor(
        edge_index[:, nonzero_idx],
        cs_scores[nonzero_idx],
        attn_size).coalesce()

    nonzero_idx = torch.nonzero(cs_scores).flatten()
    attn = torch.sparse_coo_tensor(
        edge_index,
        cs_scores,
        attn_size).coalesce()

    # min-max normalization
    if args.ATTN_NORMALIZATION == True:
        start = time.time()
        attn = sparse_min_max_norm(attn)

        if args.VERBOSE == True:
            print('Total Normalization: {:0.4f}'.format(time.time()-start))
        return attn

    return SparseTensor.from_torch_sparse_coo_tensor(attn)
