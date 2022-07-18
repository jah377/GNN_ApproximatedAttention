import time
import torch
import random

import numpy as np
from scipy import sparse

from torch_sparse import SparseTensor
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


def sparse_min_max_norm(s_coo):
    """ row-level min-max normalization """
    # https://stackoverflow.com/questions/51570512/minmax-scale-sparse-matrix-excluding-zero-elements
    assert type(s_coo) == torch.Tensor, f'{type(s_coo)} not torch.Tensor'
    assert s_coo.layout == torch.sparse_coo, f'{s_coo.type} not in torch.sparse_coo format'

    # convert to scipy.sparse.csr_matrix
    if not s_coo.is_coalesced():
        s_coo = s_coo.coalesce()
    r, c = s_coo.indices()
    v = s_coo.values()
    shape = list(s_coo.shape)
    s_csr = sparse.csr_matrix((v, (r, c)), shape=shape)

    # calculate min-max per row
    v = s_csr.data
    nnz = s_csr.getnnz(axis=1)          # total edges per row
    idx = np.r_[0, nnz[:-1].cumsum()]   # v.idx corresponding to each row

    max_r = np.maximum.reduceat(v, idx)  # max per row
    min_r = np.minimum.reduceat(v, idx)  # min per row
    min_r *= (nnz == shape[1])          # if not fully-connected, min=0

    # create matrices for vectorization
    max_m = np.repeat(max_r, nnz)
    min_m = np.repeat(min_r, nnz)

    return SparseTensor(
        row=r,
        col=c,
        value=torch.tensor((v-min_m)/(max_m-min_m)),
        sparse_sizes=shape
    )

def create_slices(dim_size, batch_size):
    """ create generator of index slices """
    count = 0
    while True:
        yield slice(count, count + int(batch_size), 1)
        count += int(batch_size)
        if count >= int(dim_size):
            break
