# %%
import copy
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

from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader


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

# pubmed: https://arxiv.org/pdf/1710.10903.pdf
# cora: https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial3/Tutorial3.ipynb?pli=1#scrollTo=FyTEd5HK8_hQ
params_dict = {
    'cora': {
        'optimizer_type': 'Adam',
        'optimizer_lr': 0.005,
        'optimizer_decay': 0.0005,
        'epochs': 100,
        'hidden_channel': 8,
        'dropout': 0.6,
        'nlayers': 2,
        'heads_in': 8,
        'heads_out': 1,
    },
    'pubmed': {
        'optimizer_type': 'Adam',
        'optimizer_lr': 0.01,
        'optimizer_decay': 0.001,
        'epochs': 100,
        'hidden_channel': 8,
        'dropout': 0.6,
        'nlayers': 2,
        'heads_in': 8,
        'heads_out': 8,
        'batch_size': 1789,
        'n_neighbors': 150,
    },
}


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


class GAT(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 hidden_channel,
                 dropout,
                 nlayers,
                 heads_in,
                 heads_out):
        """
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py


        Args:
            in_channel:         dim of features
            out_channel:        number of classes
            hidden_channel:     dim of hidden layers
            dropout:            dropout percentage
            nlayers:            total number of layers (min 2)
            heads_in:           n attention heads at INPUT and HIDDEN layers
            heads_out:          n attention heads at OUTPUT layers
        """
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel
        self.dropout = dropout
        self.nlayers = max(2, nlayers)
        self.heads_in = heads_in
        self.heads_out = heads_out

        # convs layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(
                self.in_channel,
                self.hidden_channel,
                heads=self.heads_in,
                dropout=self.dropout
            ))
        for _ in range(nlayers-2):
            self.convs.append(
                GATConv(
                    self.hidden_channel*self.heads_in,
                    self.hidden_channel,
                    heads=self.heads_in,
                    dropout=self.dropout
                ))
        self.convs.append(
            GATConv(
                self.hidden_channel*self.heads_in,
                self.out_channel,
                heads=self.heads_out,
                dropout=self.dropout,
                concat=False,
            ))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    @time_wrapper
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        for i, conv in enumerate(self.convs):
            xs = []

            for batch in subgraph_loader:
                global_ids = batch.n_id.to(x_all.device)  # expect cpu
                x = x_all[global_ids].to(device)          # expect gpu
                x = conv(x, batch.edge_index.to(x.device))

                # activation for all but final GATConv
                if i < len(self.convs) - 1:
                    x = x.relu_()

                # first :batch_size in correct order
                xs.append(x[:batch.batch_size].cpu())

            x_all = torch.cat(xs, dim=0)

        return F.log_softmax(x_all, dim=-1)

    @torch.no_grad()
    def extract_features(self, x_all, subgraph_loader):
        """ extract attention weights from trained model

        Args:
            x_all:              data.x
            subgraph_loader:    object

        Returns:
            SparseTensor  

        """

        # create storage coos for attn and edge_index count
        dim = subgraph_loader.data.num_nodes
        attn_coo = torch.sparse_coo_tensor(size=(dim, dim)).cpu()
        count_total = torch.sparse_coo_tensor(size=(dim, dim)).cpu()

        for i, conv in enumerate(self.convs):
            xs = []

            for batch in subgraph_loader:

                global_ids = batch.n_id.to(x_all.device)  # expect cpu
                x = x_all[global_ids].to(device)          # expect gpu

                if i < len(self.convs)-1:
                    x = conv(x, batch.edge_index.to(x.device))
                    x = x.relu_()
                else:
                    # extract attention weights on final GATConv layer
                    x, (attn_i, attn_w) = conv(
                        x,
                        batch.edge_index.to(x.device),
                        return_attention_weights=True
                    )

                    # store returned attn weights and indices
                    indices = global_ids[attn_i].cpu()
                    values = attn_w.mean(dim=1).detach().cpu()

                    attn_coo += torch.sparse_coo_tensor(
                        indices,
                        values,
                        size=(dim, dim)
                    )

                    count_total += torch.sparse_coo_tensor(
                        indices,
                        torch.ones_like(values),
                        size=(dim, dim)
                    )

                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)

        # average attention = attn_total / count_total
        attn_coo = attn_coo.multiply(count_total.float_power(-1)).coalesce()

        # convert to SparseTensor
        row, col = attn_coo.indices()
        values = attn_coo.values().detach()
        attn_sparse = SparseTensor(
            row=row,
            col=col,
            value=values,
            sparse_sizes=(dim, dim)
        )

        return attn_sparse


@time_wrapper
def transform_wAttention(data, K: int, params_dict):
    """
    Args:
        data:           data object
        K:              number of SIGN transformations
        attn_heads:     number of attention heads

    Returns:
        data:   transformed data
        time_:  from wrapper
        mem_:   from wrapper
    """

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
    # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    # replace adj with Cosine Similarity weights
    adj_t = extract_attention(data, params_dict.get(args.dataset))
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    # sanity check
    if K > 0:
        assert hasattr(data, f'x{K}')

    return data


def extract_attention(data, GATdict):
    """ train GAT model and extract attention
    Args:
        data:
        GATtransform_params:     number of attention heads

    Returns:
        SparseTensor containing attention weights
    """

    # CREATE TRAINING AND SUBGRAPH LOADERS
    # [n_neighbors] = hyperparameter
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,  # can be bool or n_id indices
        num_neighbors=[GATdict['n_neighbors']]*GATdict['nlayers'],
        shuffle=True,
        batch_size=GATdict['batch_size'],
        drop_last=True,  # remove final batch if incomplete
    )

    subgraph_loader = NeighborLoader(
        copy.copy(data),
        input_nodes=None,
        num_neighbors=[-1]*GATdict['nlayers'],  # sample all neighbors
        shuffle=False,                          # :batch_size in sequential order
        batch_size=GATdict['batch_size'],
        drop_last=False,
    )
    subgraph_loader.data.num_nodes = data.num_nodes
    del subgraph_loader.data.x, subgraph_loader.data.y  # only need indices

    # BUILD MODEL
    model = GAT(
        data.num_features,  # in_channel
        data.num_classes,  # out_channel
        GATdict['hidden_channel'],
        GATdict['dropout'],
        GATdict['nlayers'],
        GATdict['heads_in'],
        GATdict['heads_out'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # BUILD OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GATdict['optimizer_lr'],
        weight_decay=GATdict['optimizer_decay'],
    )

    # BUILD SCHEDULER (modulates learning rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',     # nll_loss expected to decrease over epochs
        factor=0.1,     # lr reduction factor
        patience=5,     # reduce lr after _ epochs of no improvement
        min_lr=1e-6,    # min learning rate
        verbose=False,  # do not monitor lr updates
    )

    # RUN THROUGH EPOCHS
    # params for early termination
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    for epoch in range(GATdict['epochs']):

        train_out, train_resources = training_step(
            model,
            optimizer,
            train_loader
        )

        test_out = testing_step(model, data, subgraph_loader)

        val_loss = test_out['val_loss']
        scheduler.step(val_loss)

        # log results
        s = 'precomp-epoch'
        log_dict = {f'{s}': epoch}
        log_dict.update({f'{s}-'+k: v for k, v in train_out.items()})
        log_dict.update({f'{s}-train-'+k: v for k,
                        v in train_resources.items()})
        log_dict.update({f'{s}-'+k: v for k, v in test_out.items()})
        wandb.log(log_dict)

        # early stopping
        current_loss = val_loss
        if current_loss > previous_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('~~~ early stop triggered ~~~')
                break
        else:
            trigger_times = 0
        previous_loss = current_loss

    # EXTRACT ATTENTION MATRIX
    model.eval()
    attn_sparse = model.extract_features(data.x, subgraph_loader)

    return attn_sparse


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

        data.y = data.y.flatten()
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

        batch_size = int(idx.numel())
        total_examples += int(batch_size)
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())

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
        batch_size = int(idx.numel())
        total_time += out_time
        total_examples += int(batch_size)
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())

    return {
        'f1': total_correct/total_examples,
        'loss': total_loss/total_examples,
        'time': total_time,  # total inference time
    }


def main(args):
    assert args.dataset in [
        'cora', 'pubmed'], f'GAT transformation unavailable for {args.dataset.title()}'
    set_seeds(args.seed)

    # data
    data = download_data(args.dataset, K=args.K)
    data = standardize_data(data, args.dataset)
    data, transform_time = transform_wAttention(
        data,
        args.dataset,
        args.K,
        params_dict.get(args.dataset),
    )

    train_loader = create_loader(
        data, split='train', batch_size=args.batch_size)
    val_loader = create_loader(data, split='val', batch_size=args.batch_size)
    test_loader = create_loader(data, split='test', batch_size=args.batch_size)

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
                'run': run, 'transform_time': transform_time, 'epoch': epoch,
                'n_params': n_params, 'training_time': training_time
            }
            epoch_dict.update(
                {f'training_{k}': v for k, v in training_out.items()})
            epoch_dict.update(
                {f'eval_train_{k}': v for k, v in train_out.items()})
            epoch_dict.update(
                {f'eval_val_{k}': v for k, v in val_out.items()})
            epoch_dict.update(
                {f'eval_test_{k}': v for k, v in test_out.items()})

            store_run = pd.concat(
                [store_run, pd.DataFrame.from_dict([epoch_dict])],
                ignore_index=True
            )

    return store_run.to_csv(
        f'{args.dataset}_output.csv',
        sep=',',
        header=True,
        index=False
    )


if __name__ == '__main__':
    main(args)
