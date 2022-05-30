# %%
import os
import os.path as osp

import torch
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset

def download_k0(data_name):
    """ download data from dataset name """

    possible_datasets = ['cora', 'pubmed', 'products', 'arxiv']
    assert data_name.lower() in possible_datasets

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToUndirected(),
        T.AddSelfLoops(),
    ])

    if data_name.lower() in ['products', 'arxiv']:
        dataset = PygNodePropPredDataset(
            f'ogbn-{data_name.lower()}',
            root=f'{data_name.title}',
            transform=transform)
    else:
        dataset = Planetoid(
            root=f'/tmp/{data_name.title()}',
            name=f'{data_name.title()}',
            transform=transform,
            split='full',
        )

    return dataset


def standardize_data(dataset, data_name: str):
    """ standardize format of data object """

    possible_datasets = ['cora', 'pubmed', 'products', 'arxiv']
    assert data_name.lower() in possible_datasets

    # extract relevant information
    data = dataset[0]
    data.dataset_name = data_name.lower()
    data.num_classes = dataset.num_classes
    data.num_nodes = data.num_nodes
    data.num_edges = data.num_edges
    data.num_node_features = data.num_node_features
    data.n_id = torch.arange(data.num_nodes)  # global node id

    # standardize mask -- node idx, not bool mask
    if hasattr(dataset, 'get_idx_split'):
        masks = dataset.get_idx_split()
        data.train_mask = masks['train']
        data.val_mask = masks['valid']
        data.test_mask = masks['test']

        data.y = data.y.flatten()
    else:
        data.train_mask = torch.where(data.train_mask)[0]
        data.val_mask = torch.where(data.val_mask)[0]
        data.test_mask = torch.where(data.test_mask)[0]

    return data


def transform_data(data, K: int=0):
    """ perform SIGN transformation """
    assert data.edge_index is not None
    row, col = data.edge_index
    adj_t = SparseTensor(row=col, col=row,
                            sparse_sizes=(data.num_nodes, data.num_nodes))

    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    assert data.x is not None
    xs = [data.x]
    for i in range(1, K + 1):
        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    return data


# %% DOWNLOAD
data_name = 'products'
parent_dirpath = os.getcwd()
path = osp.join(parent_dirpath, 'data', data_name)

if not osp.exists(path):
    os.makedirs(path)

dataset = download_k0(data_name)

#%% STANDARDIZE 

data = standardize_data(dataset, data_name)

# %% TRANSFORM

max_K = 6

# transform
for K in range(max_K):
    torch.save(
        transform_data(data, K=K),
        osp.join(path, f'{data_name}_sign_k{K}.pth'),
        )
    print(f'{data_name.upper()} K={K} complete! ')

# %%
