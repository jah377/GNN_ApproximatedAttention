# %%
import time
import argparse
import random
import numpy as np
import pandas as pd
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset


# product: https://arxiv.org/pdf/2004.11198v2.pdf
parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='products')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--optimizer_lr', type=float, default=0.0001)
parser.add_argument('--optimizer_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--hidden_channel', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--batch_norm', type=strtobool, default=True)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SIGN(torch.nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 hidden_channel,
                 dropout,
                 K,
                 batch_norm):
        """
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sign.py

        Args:
            in_channel:         dim of features
            out_channel:        number of classes
            hidden_channel:     dim of hidden layers
            dropout:            dropout percentage
            K:                  SIGN hyperparameter
            batch_norm:         if True, perform batch normalization
        """
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel
        self.dropout = dropout
        self.K = K
        self.batch_norm = batch_norm

        # create linear submodel to be used on AX, ..., A^KX data
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.K + 1):
            self.lins.append(
                nn.Linear(self.in_channel, self.hidden_channel))
            self.bns.append(
                nn.BatchNorm1d(self.hidden_channel))

        # create linear submodel to be used concat data
        self.lin = nn.Linear(
            (self.K + 1) * self.hidden_channel, self.out_channel)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, xs):
        """ xs = [AX^0, AX^1, ..., AX^K] """
        hs = []  # store forward pass of each AX^K

        for i, lin in enumerate(self.lins):
            h = lin(xs[i])              # linear transform
            h = self.bns[i](h) \
                if self.batch_norm \
                else h                  # batch norm
            h.relu()                    # activation
            h = F.dropout(h, self.dropout, training=self.training)
            hs.append(h)

        h = torch.cat(hs, dim=-1)
        h = self.lin(h)

        return h.log_softmax(dim=-1)    # calc final predictions


def time_wrapper(func):
    """wrapper for recording time
    https://gmpy.dev/blog/2016/real-process-memory-and-environ-in-python
    https://psutil.readthedocs.io/en/latest/index.html?highlight=virtual%20memory#psutil.virtual_memory

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
        eval(f'data.{split}_mask'),
        batch_size=batch_size,
        shuffle=(split == 'train'),   # shuffle if training loader
        drop_last=(split == 'train'),  # remove final incomplete
        num_workers=num_workers,
    )


@time_wrapper
def train_epoch(model, data, optimizer, loader):
    """ Perform forward and backward pass on SIGN
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sign.py
    Args:
        model:      SIGN model
        data:       data object
        loader:     DataLoader of train/val/test set
        optimizer:  optimizer object

    Returns:
        train_loss:     loss @ epoch
        train_f1:       f1 @ epoch
        delta_time:     from wrapper
    """
    model.train()

    total_examples = total_correct = total_loss = 0
    for idx in loader:

        # organize data
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # forward pass
        out = model(xs)
        loss = F.nll_loss(out, y)

        batch_size = idx.numel()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_correct += sum(out.argmax(dim=-1) == y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        'f1': total_correct/total_examples,
        'loss': total_loss/total_examples,
    }


@torch.no_grad()
def test_epoch(model, data, loader):
    """ Document validation or test loss and accuracy
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sign.py

    Args:
        model:      trained GAT model
        data:       data object
        loader:     train, val, or test DataLoader

    Returns:
        loss:       loss @ epoch
        f1:         f1 @ epoch
        delta_time:     from wrapper
    """
    model.eval()

    @time_wrapper
    def predict(model, xs):
        return model(xs)

    total_time = total_loss = 0
    total_examples = total_correct = 0

    for idx in loader:

        # organize data
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # forward pass
        out, out_time = predict(model, xs)
        loss = F.nll_loss(out, y)

        # store
        batch_size = idx.numel()
        total_time += out_time
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_correct += sum(out.argmax(dim=-1) == y)

    return {
        'f1': total_correct/total_examples,
        'loss': total_loss/total_examples,
        'time': total_time,  # total inference time
    }

# %%


def main(args):
    set_seeds(args.seed)

    # data
    data = download_data(args.dataset, K=args.K)
    data = standardize_data(data, args.dataset)

    train_loader = create_loader(data, 'train', batch_size=args.batch_size)
    val_loader = create_loader(data, 'val', batch_size=args.batch_size)
    test_loader = create_loader(data, 'test', batch_size=args.batch_size)

    # model
    model = SIGN(
        data.num_features,       # in_channel
        data.num_classes,        # out_channel
        args.hidden_channel,
        args.dropout,
        args.K,
        args.batch_norm).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    store_run = pd.DataFrame()
    for run in range(args.n_runs):
        model.reset_parameters

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.optimizer_lr,
            weight_decay=args.optimizer_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',     # nll_loss expected to decrease over epochs
            factor=0.1,     # lr reduction factor
            patience=5,     # reduce lr after _ epochs of no improvement
            min_lr=1e-6,    # min learning rate
            verbose=False,  # do not monitor lr updates
        )

        for epoch in range(args.epochs):

            training_out, training_time = train_epoch(
                model, data, optimizer, train_loader)

            train_out = test_epoch(model, data, train_loader)
            val_out = test_epoch(model, data, val_loader)
            test_out = test_epoch(model, data, test_loader)

            scheduler.step(val_out['loss'])

            # store epoch
            epoch_dict = {
                'run': run, 'epoch': epoch,
                'n_params': n_params, 'training_time': training_time
            }
            epoch_dict.update({f'training_{k}': v for k, v in training_out.items()})
            epoch_dict.update({f'eval_train_{k}': v for k, v in train_out.items()})
            epoch_dict.update({f'eval_val_{k}': v for k, v in val_out.items()})
            epoch_dict.update({f'eval_test_{k}': v for k, v in test_out.items()})

            store_run = pd.concat(
                [store_run, pd.DataFrame.from_dict([epoch_dict])],
                ignore_index=True
            )

    return store_run.to_csv(
        f'{args.dataset}_SIGN.csv',
        sep=',',
        header=True,
        index=False
    )


if __name__ == '__main__':
    main(args)
