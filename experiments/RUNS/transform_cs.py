
import time
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

    cs_times = np.empty(num_edges)

    for i in range(num_edges):
        start = time.time()
        A, B = x[edge_index[:, i]].to(device)
        score = F.cosine_similarity(A, B, dim=-1)
        cs_scores[i] = score.cpu()

        cs_times[i] = time.time()-start

        del A, B, score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('For-Loop Times: {:0.4f}+\-{:0.4f} [{:0.4f}-{:0.4f}]'.format(
        cs_times.mean(), cs_times.std(), cs_times.min(), cs_times.max()))

    # remove idx of cs_score
    nonzero_idx = np.nonzero(cs_scores != 0)[0]
    attn = torch.sparse_coo_tensor(
        edge_index[:, nonzero_idx],
        torch.from_numpy(cs_scores[nonzero_idx]),
        attn_size)

    # min-max normalization
    if args.ATTN_NORMALIZATION == True:
        start = time.time()
        attn = sparse_min_max_norm(attn)
        print('Normalization Time: {:0.4f}'.format(time.time()-start))
        return attn

    return SparseTensor.from_torch_sparse_coo_tensor(attn)
