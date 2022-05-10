import torch
from torch_sparse import SparseTensor

from general.models.DotProductAttention import net as MultiheadAttention
from general.utils import resources  # wrapper


@resources
def transform_wAttention(data, K: int, attn_heads: int = 1, mha_bias: int = 1):
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
    adj_t = extract_attention(data.x, data.edge_index, attn_heads, mha_bias)
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    return data


def extract_attention(x, edge_index, attn_heads, mha_bias):
    """ calculate dotproduct attention
    Args:
        x:              feature embeddings [n nodes x emb]
        edge_index:     connections
        attn_heads:     number of attention heads
        mha_bias:       learn additive bias (default=True) 

    Returns:
        torch.sparse_coo_matrix of attention weights
    """

    d_m = x.shape[1]  # feature embedding dimension
    MHA = MultiheadAttention(d_m, attn_heads, mha_bias)

    return MHA(x, edge_index)
