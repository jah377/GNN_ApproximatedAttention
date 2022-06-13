import torch
from torch_sparse import SparseTensor

from utils import time_wrapper
from SamplerGAT_configs import GATparams
from transform_cs import cosine_filter
from transform_dp import dotproduct_filter
from transform_gat import gat_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@time_wrapper
def transform_data(data, args):
    """ SIGN transformation with attention filter """
    assert hasattr(
        args, 'TRANSFORMATION'), "Must specify TRANSFORMATION ['gat', 'cosine', 'dot_product', 'cosine_per_k']"

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

    # replace adj_t with attention filter
    if args.TRANSFORMATION.lower() == 'gat':
        assert args.DATASET.lower() in [
            'cora', 'pubmed'], f'GAT transformation unavailable for {args.DATASET}'
        adj_t = gat_filter(data, args, GATparams.get(args.DATASET.lower()))

    elif args.TRANSFORMATION.lower() == 'cosine':
        adj_t = cosine_filter(data.x0, data.edge_index, args)

    elif args.TRANSFORMATION.lower() == 'dot_product':
        adj_t = dotproduct_filter(data, args)

    elif args.TRANSFORMATION.lower() == 'cosine_per_k':
        assert data.x0 is not None
        xs = [data.x0]

        for i in range(1, args.K + 1):
            adj_t = cosine_filter(xs[-1], data.edge_index, args)
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            xs += [adj_t @ xs[-1]]
            data[f'x{i}'] = xs[-1]
            del adj_t

        return data

    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    assert data.x0 is not None
    xs = [data.x0]

    for i in range(1, args.K + 1):
        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    return data
