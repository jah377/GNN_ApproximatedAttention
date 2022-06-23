# %%
import time
import glob
import torch
import random

import numpy as np
from scipy import sparse

from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from ogb.nodeproppred import Evaluator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(dataset):
    """ load original data (K=0) from dataset name """

    file_name = f'{dataset}_sign_k0.pth'
    path = glob.glob(f'./**/{file_name}', recursive=True)[0][2:]
    return torch.load(path)

def create_evaluator_fn(dataset: str):
    """ create function to determine accuracy score """

    if dataset in ['arxiv', 'products']:
        evaluator = Evaluator(name=f'ogbn-{dataset}')
        return lambda preds, labels: evaluator.eval({
            'y_true': labels.view(-1, 1),
            'y_pred': preds.view(-1, 1),
        })['acc']

    return lambda preds, labels: (preds == labels).numpy().mean()

def parse_data(data, dataset, hops):
    """ parse Data object for relevant variables """
    xs = [data.x]+[data[f'x{k}'].float() for k in range(1, hops+1)]
    labels = data.y.to(device)
    n_features = data.num_features
    n_classes = data.num_classes
    train_idx = data['train_mask']
    val_idx = data['val_mask']
    test_idx = data['test_mask']
    evaluator = create_evaluator_fn(dataset)

    return [xs, labels, n_features, n_classes, train_idx, val_idx, test_idx, evaluator]

def transform_data(data, hops):
    """ SIGN transformation with or without attention filter """
    xs = [data.x]  # store transformations

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
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    for i in range(1, hops + 1):
        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    return data

# %%
dataset = 'arxiv'
hops = 5
batch_size=50000
eval_batch_size=100000

data = load_data(dataset)
data = transform_data(data, hops)
data = parse_data(data,dataset, hops)
xs, labels, n_features, n_classes, train_idx, val_idx, test_idx, evaluator = data

train_loader = torch.utils.data.DataLoader(
    train_idx,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

test_loader = torch.utils.data.DataLoader(
    torch.from_numpy(np.arange(labels.shape[0])),
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False
)

# %%
for batch in test_loader:
    batch_feats = [feat[batch] for feat in xs]
# %%
