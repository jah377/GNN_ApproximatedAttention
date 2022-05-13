import time
import math
import copy
import argparse
import numpy as np
import os.path as osp
from einops import rearrange, reduce, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

from torch_sparse import SparseTensor


from general.utils import standardize_dataset

parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='pubmed')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    # IMPORT & STANDARDIZE DATA
    path = f'data/{args.dataset}_sign_k0.pth'
    dataset = torch.load(path)
    data = standardize_dataset(dataset, 'pubmed')
    del dataset

    # CREATE TRAINING AND SUBGRAPH LOADERS
    # [n_neighbors] = hyperparameter
    n_neighbors = -1  # sample __ neighbors for each node, at each layer
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,  # can be bool or n_id indices
        num_neighbors=[n_neighbors]*GATtransform_params['nlayers'],
        shuffle=True,
        batch_size=GATtransform_params['batch_size'],
        drop_last=True,  # remove final batch if incomplete
    )

    n_neighbors = -1  # sample all neighbors
    subgraph_loader = NeighborLoader(
        copy.copy(data),
        input_nodes=None,
        num_neighbors=[n_neighbors]*GATtransform_params['nlayers'],
        shuffle=False,  # :batch_size = primary batch nodes, rest are neighbors
        batch_size=GATtransform_params['batch_size'],
    )
    subgraph_loader.data.num_nodes = data.num_nodes
    del subgraph_loader.data.x, subgraph_loader.data.y  # only need indices

    # BUILD MODEL
    model = net(
        data.num_features,  # in_channel
        data.num_classes,  # out_channel
        GATtransform_params['hidden_channel'],
        GATtransform_params['dropout'],
        GATtransform_params['nlayers'],
        GATtransform_params['heads_in'],
        GATtransform_params['heads_out'],
    ).to(device)

    # BUILD OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GATtransform_params['optimizer_lr'],
        weight_decay=GATtransform_params['optimizer_decay'],
    )

    # BUILD SCHEDULER (modulates learning rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',   # nll_loss expected to decrease over epochs
        factor=0.1,   # lr reduction factor
        patience=5,   # reduce lr after _ epochs of no improvement
        min_lr=1e-6,  # min learning rate
        verbose=False,  # do not monitor lr updates
    )

    # RUN THROUGH EPOCHS
    # params for early termination
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    # train and validation
    for epoch in range(GATtransform_params['epochs']):

        training_step(model, optimizer, train_loader)

        loss, acc, _ = testing_step(model, optimizer, data, subgraph_loader)

        scheduler.step(loss['val_loss'])

        # early stopping
        current_loss = loss['val_loss']
        if current_loss > previous_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('~~~ early stop triggered ~~~')
                break
        else:
            trigger_times = 0
        previous_loss = current_loss

        print(f'Epoch: {epoch:02d}, Train: {acc['train']:.4f}, Val: {acc['val']:.4f}, '
              f'Test: {acc['test']:.4f}')
