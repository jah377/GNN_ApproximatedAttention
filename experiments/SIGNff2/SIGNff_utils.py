import time 
import torch
import numpy as np
import random 
from torch.utils.data import DataLoader
from ogb.nodeproppred import Evaluator

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


def create_evaluator_fn(dataset: str):
    """ create function to determine accuracy score """

    if dataset in ['arxiv', 'products']:
        evaluator = Evaluator(name=f'ogbn-{dataset}')
        return lambda preds, labels: evaluator.eval({
            'y_true': labels.view(-1, 1),
            'y_pred': preds.view(-1, 1),
        })['acc']
    else:
        return lambda preds, labels: (preds == labels).numpy().mean()


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