import argparse
import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset

parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--data_name', type=str, default='pubmed')
args = parser.parse_args()


def download_data(data_name, K: int = 0):
    """ download data from name str """

    possible_datasets = ['cora', 'pubmed', 'products', 'arxiv']
    assert data_name.lower() in possible_datasets
    assert isinstance(K, int)

    if K == 0:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
        ])
    else:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
            T.SIGN(K)
        ])

    if data_name.lower() in ['products', 'arxiv']:
        dataset = PygNodePropPredDataset(
            f'ogbn-{data_name.lower()}',
            root=f'/tmp/{data_name.title}',
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
    """ standardize format out data object """

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
    if data_name.lower() in ['products', 'arxiv']:
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


def percent_split(data):
    n_train = data.train_mask.shape[0]
    n_val = data.val_mask.shape[0]
    n_test = data.test_mask.shape[0]
    n_total = n_train+n_val+n_test

    return {
        'train': n_train/n_total,
        'val': n_val/n_total,
        'test': n_test/n_total,
    }


def total_edges(data):
    return data.num_edges if data.is_directed() else np.ceil(data.num_edges/2)


def average_degree(data):
    coo = torch.sparse_coo_tensor(
        indices=data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_nodes),
    )

    edges_per_node = torch.sparse.sum(coo, dim=-1).values()
    del coo

    avg_deg = edges_per_node.mean()
    std_deg = edges_per_node.std()
    max_deg = edges_per_node.max()
    min_deg = edges_per_node.min()

    return float(avg_deg), float(std_deg), float(max_deg), float(min_deg)


def homophily_degree(data):
    r, c = data.edge_index
    y = data.y
    return sum(y[r] == y[c])/len(r)


def get_statistics(args):
    data_name = args.data_name

    data = download_data(data_name)
    # data = torch.load(f'data/{data_name}/{data_name}_sign_k0.pth')

    data = standardize_data(data, data_name)
    splits = percent_split(data)
    n_edges = total_edges(data)
    avg_deg, std_deg, max_deg, min_deg = average_degree(data)
    homo_deg = homophily_degree(data)

    print()
    print(f'\n==== {data_name.upper()} Statistics =====')
    print('Nodes: {:,}'.format(data.num_nodes))
    print('Edges: {:,}'.format(n_edges))
    print('Features: {:,}'.format(data.num_features))
    print('Classes: {:,}'.format(data.num_classes))
    print('Directed?: {:}'.format(data.is_directed()))
    print('%Train: {}'.format(round(splits['train'], 3)))
    print('%Val: {}'.format(round(splits['val'], 3)))
    print('%Test: {}'.format(round(splits['test'], 3)))
    print('Avg. Deg.: {}+/-{} [{}-{}]'.format(
        round(avg_deg, 3),
        round(std_deg, 3),
        round(min_deg, 3),
        round(max_deg, 3),
    ))
    print('AVG DEG SANITY CHECK: {}'.format(data.num_edges/data.num_nodes))
    print('Homophily Deg.: {}'.format(round(homo_deg.item(), 3)))
    print()

    del data


if __name__ == '__main__':
    get_statistics(args)
