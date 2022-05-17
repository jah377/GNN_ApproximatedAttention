import time
import psutil
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

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


def standardize_dataset(data_obj, data_str):
    assert data_str.lower() in ['cora', 'pubmed', 'products', 'arxiv']

    # extract relevant information
    data = data_obj[0]
    data.num_classes = data_obj.num_classes
    data.num_nodes = data.num_nodes
    data.num_features = data.num_node_features
    data.n_id = torch.arange(data.num_nodes)  # global node id

    # standardize mask -- node idx, not bool mask
    if data_str.lower() in ['products', 'arxiv']:
        masks = data_obj.get_idx_split()
        data.train_mask = masks['train']
        data.val_mask = masks['valid']
        data.test_mask = masks['test']

        data.y = data.y.flatten() # original: [n,1]
    else:
        data.train_mask = torch.where(data.train_mask)[0]
        data.val_mask = torch.where(data.val_mask)[0]
        data.test_mask = torch.where(data.test_mask)[0]

    return data


def build_DataLoader(data, batch_size: int):
    """ Create train/val/test DataLoader for SIGN

    Args:
        idx:            all idx values
        batch_size:     int, batch_size

    Returns:
        train_loader:
        val_loader:
        test_loader:
    """

    def loader(data, split):
        return DataLoader(
            eval(f'data.{split}_mask'),
            batch_size=batch_size,
            shuffle=(split == 'train'),   # shuffle if training loader
            drop_last=(split == 'train'),  # remove final incomplete
        )

    return [loader(data, split) for split in ['train', 'val', 'test']]
