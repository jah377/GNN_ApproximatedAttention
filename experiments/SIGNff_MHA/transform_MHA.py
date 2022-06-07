import torch
from torch_sparse import SparseTensor

from model_DPA import DotProductAttention
from general.utils import time_wrapper  # wrapper


@time_wrapper
def DPAttention(data, K: int, attn_heads: int = 1, norm: bool = True):
    """
    Args:
        data:           data object
        K:              number of SIGN transformations
        attn_heads:     number of attention heads

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
    adj_t = extract_attention(data, attn_heads, norm)
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


def extract_attention(data, attn_heads, norm: bool = True):
    """ calculate dotproduct attention
    Args:
        data:
        attn_heads:     number of attention heads

    Returns:
        SparseTensor containing attention weights
    """

    model = DotProductAttention(
        data.num_nodes,
        data.num_features,
        data.num_edges,
        attn_heads,
        norm,
    )

    return model(data.x, data.edge_index)
