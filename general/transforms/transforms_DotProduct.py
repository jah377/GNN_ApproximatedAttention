import torch
from torch_sparse import SparseTensor

from general.models.DotProductAttention import net as MultiheadAttention
from general.utils import resources  # wrapper


@resources
def transform_wAttention(data, K: int, attn_heads: int = 1):
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
    adj_t = extract_attention(data, attn_heads)
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    return data


def extract_attention(data, attn_heads):
    """ calculate dotproduct attention
    Args:
        x:              feature embeddings [n nodes x emb]
        edge_index:     connections
        attn_heads:     number of attention heads
        mha_bias:       learn additive bias (default=True) 

    Returns:
        SparseTensor containing attention weights
    """

    model = MultiheadAttention(data.num_features, attn_heads)

    return model(data.x, data.edge_index)
