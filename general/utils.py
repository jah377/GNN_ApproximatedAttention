import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def time_wrapper(func):
    """ wrapper for recording time
    Args:
        func:   function to evaluate

    Return:
        output:         output of func
        delta_time:     seconds, time to exec func
    """
    def wrapper(*args, **kwargs):

        time_initial = time.time()
        output = func(*args, **kwargs)
        time_end = time.time()-time_initial

        # unpack tuple if func returns multiple outputs
        if isinstance(output, tuple):
            return *output, time_end

        return output, time_end

    return wrapper


def set_seeds(seed_value: int):
    """ for reproducibility """

    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


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


@time_wrapper
def transform_data(data, K: int = 0):
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


def create_loader(data, split: str, batch_size: int, num_workers: int = 1):
    """ build DataLoader object based on inputs """

    assert split in ['train', 'val', 'test', 'all']

    return DataLoader(
        data.n_id if split == 'all' else data[f'{split}_mask'],
        batch_size=batch_size,
        shuffle=(split == 'train'),   # shuffle if training loader
        drop_last=(split == 'train'),  # remove final incomplete
        num_workers=num_workers,
    )


def get_n_params(model):
    """
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/sign/sign.py
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
