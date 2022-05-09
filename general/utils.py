import time
import psutil
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

        delta_memory = memory() - memory_initial  # bytes
        delta_time = time.time() - time_initial         # seconds

        return output, delta_time, delta_memory

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


# def get_data(dataset_name, transform=None):
#     dataset_name = dataset_name.lower()
#     assert dataset_name in ['cora', 'pubmed']

#     path = os.path.join(os. getcwd(), 'data', 'test')

#     if dataset_name == 'pubmed':
#         datamodule = Planetoid(root=path,
#                                name='PubMed',
#                                split='full',
#                                transform=transform)[0]
#     elif dataset_name == 'cora':
#         datamodule = Planetoid(root=path,
#                                name='Cora',
#                                split='full',
#                                transform=transform)[0]
#     return datamodule


def build_optimizer(model, name: str, learning_rate: float, weight_decay: float):
    """ build optimizer
    Args:
        model:              model object
        name:               name of optimizer
        learning_rate:      
        weight_decay:
    Return:
        optimizer object
    """
    name = name.title()
    assert name in ['Adam']

    if name == 'Adam':
        optimizer = eval(f'torch.optim.{name}')

    return optimizer(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer):
    """ build learning rate scheduler
    Args:
        optimizer:  object

    Return:
        scheduler:  object
    """
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        min_lr=1e-6,
        verbose=False
    )


def build_DataLoader(data, batch_size: int):
    """ Create train/val/test DataLoader for SIGN

    Args:
        data:           data object
        batch_size:     int, batch_size

    Returns:
        train_loader:
        val_loader:
        test_loader:
    """

    def get_idx(data, split):
        return eval(
            f"data.{split}_mask.nonzero(as_tuple=False).view(-1)"
        )

    def loader(data, split):

        return DataLoader(
            get_idx(data, split),
            batch_size=batch_size,
            shuffle=(split == 'train'),     # shuffle if training loader
            drop_last=(split == 'train'),   # remove the final incomplete batch 
        )

    return [loader(data, split) for split in ['train', 'val', 'test']]
