import wandb
import copy

import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from general.utils import resources  # wrapper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@resources
def transform_wAttention(data, K: int, batch_size: int):
    """
    Args:
        data:           data object
        K:              number of SIGN transformations
        GATtransform_params:     dict of best hyperparameters determined from sweep

    Returns:
        data:   transformed data
        time_:  from wrapper
        mem_:   from wrapper
    """

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
    # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    # replace adj with DotProductAttention weights
    adj_t = extract_attention(data, K, batch_size)
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    # sanity check
    if K > 0:
        assert hasattr(data, f'x{K}')

    return data


# def extract_attention(data, K, batch_size):
#     """ calculate dotproduct attention
#     Args:
#         data:           torch data object
#         K:              technically the number of layers
#         batch_size:     use SIGN hyperparameter

#     Returns:
#         SparseTensor containing cosine similarity weights
#     """

#     # sample neighboring nodes to compute similarities
#     loader = NeighborLoader(
#         copy.copy(data),
#         input_nodes=None,
#         num_neighbors=[-1]*K,  # include all neighbors
#         shuffle=False,
#         batch_size=batch_size,
#     )
#     del loader.data.x, loader.data.y  # only need indices

#     # store cumulative 'similarity' and number of occurances per edge_index
#     x = data.x
#     dim = data.num_nodes
#     cs_coo = torch.sparse_coo_tensor(size=(dim, dim)).cpu()
#     count_total = torch.sparse_coo_tensor(size=(dim, dim)).cpu()

#     for batch in loader:
#         r, c = batch.edge_index
#         values = F.cosine_similarity(
#             x[batch.n_id[r]].to(device),
#             x[batch.n_id[c]].to(device)
#         )

#         cs_coo += torch.sparse_coo_tensor(
#             batch.edge_index,
#             values.cpu(),
#             size=(dim, dim)
#         )

#         count_total += torch.sparse_coo_tensor(
#             batch.edge_index,
#             torch.ones_like(values).cpu(),
#             size=(dim, dim)
#         )

#     # average CosineSimilarity per edge_index
#     cs_coo = cs_coo.multiply(count_total.float_power(-1)).coalesce().cpu()

#     # convert to SparseTensor
#     row, col = cs_coo.indices()
#     cs_sparse = SparseTensor(
#         row=row,
#         col=col,
#         value=cs_coo.values(),
#         sparse_sizes=(dim, dim)
#     )

#     return cs_sparse


def extract_attention(x, edge_index, cs_batch_size):
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]

    def _batch_slices(num_edges, batch_size):
        """Generator that yields slice objects for indexing into 
        sequential blocks of an array along a particular axis
        """
        count = 0
        while True:
            yield slice(count, count + int(batch_size), 1)
            count += int(batch_size)
            if count >= int(num_edges):
                break

    values = torch.tensor(range(num_edges)).cpu()

    for batch in _batch_slices(num_edges, cs_batch_size):
        edges = edge_index[:, batch]  # edge_idx -> node_idx
        A = x[edges[0]].to(device)
        B = x[edges[1]].to(device)
        values[batch] = F.cosine_similarity(A, B, dim=1).cpu()

        del A, B
        if torch.cuda.is_available():
            torch.cuda.empty()

    return SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=values,
        sparse_sizes=(num_nodes, num_nodes)
    )
