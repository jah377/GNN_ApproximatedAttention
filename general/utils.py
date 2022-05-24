import time
import psutil
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resources(func):
    # TODO memory consumption incorrect
    """wrapper for recording time and memory consumption
    https://gmpy.dev/blog/2016/real-process-memory-and-environ-in-python
    https://psutil.readthedocs.io/en/latest/index.html?highlight=virtual%20memory#psutil.virtual_memory

    Args:
        func:   function to evaluate

    Return:
        output:         output of func
        delta_time:     seconds, time to exec func
        delta_memory:   bytes, memory consumed by func
    """

    def memory():

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(device)

        return psutil.virtual_memory().available

    def wrapper(*args, **kwargs):

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        memory_initial = memory()
        time_initial = time.time()

        output = func(*args, **kwargs)

        resource_dict = {
            'time': time.time() - time_initial,  # seconds,
            'mem':  memory() - memory_initial,  # bytes
        }

        # unpack tuple if func returns multiple outputs
        if isinstance(output, tuple):
            return *output, resource_dict

        return output, resource_dict

    return wrapper


def set_seeds(seed_value: int):
    """ Set seeds across modules
    Args:
        seed_value:     int, desired seed value
    """
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def download_data(data_name, K: int = 0):
    possible_datasets = ['cora', 'pubmed', 'products', 'arxiv']
    assert data_name.lower() in possible_datasets
    assert isinstance(K, int)

    if K == 0:
        transform = T.NormalizeFeatures()
    else:
        transform = T.Compose([
            T.NormalizeFeatures(),
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
    possible_datasets = ['cora', 'pubmed', 'products', 'arxiv']
    assert data_name.lower() in possible_datasets

    # extract relevant information
    data = dataset[0]
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
    else:
        data.train_mask = torch.where(data.train_mask)[0]
        data.val_mask = torch.where(data.val_mask)[0]
        data.test_mask = torch.where(data.test_mask)[0]

    return data


def create_loader(data, split: str, batch_size: int, num_workers: int = 1):
    assert split in ['train', 'val', 'test']

    return DataLoader(
        data[f'{split}_mask'],
        batch_size=batch_size,
        shuffle=(split == 'train'),   # shuffle if training loader
        drop_last=(split == 'train'),  # remove final incomplete
        num_workers=num_workers,
    )
