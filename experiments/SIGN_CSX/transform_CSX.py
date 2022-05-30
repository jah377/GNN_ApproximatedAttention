import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F

from general.utils import time_wrapper  # wrapper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@time_wrapper
def CosineAttention_eachX(data, K: int, cs_batch_size: int = 1):
    """ calculate CS for each K
    Args:
        data:           data object
        K:              number of SIGN transformations
        cs_batch_size:  CosSim batch size (n nodes)

    Returns:
        data:           transformed data
        time_:          from wrapper
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

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        adj_t = extract_attention(xs[-1], data.edge_index, cs_batch_size)
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]
        del adj_t

    # sanity check
    if K > 0:
        assert hasattr(data, f'x{K}')

    return data


def create_slices(dim_size, batch_size):
    """ create generator of index slices """

    count = 0
    while True:
        yield slice(count, count + int(batch_size), 1)
        count += int(batch_size)
        if count >= int(dim_size):
            break


def extract_attention(x, edge_index, cs_batch_size, ):
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]

    values = torch.tensor(range(num_edges)).cpu()

    for batch in create_slices(num_edges, cs_batch_size):
        A = x[edge_index[0, batch]].to(device)
        B = x[edge_index[1, batch]].to(device)
        values[batch] = F.cosine_similarity(A, B, dim=1).cpu()

        del A, B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=values,
        sparse_sizes=(num_nodes, num_nodes)
    )
