import time
import copy
import random
import argparse
import numpy as np
import pandas as pd
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GATConv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from torch_sparse import SparseTensor
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
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def time_wrapper(func):
    """wrapper for recording time

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
def train_epoch(model, optimizer, loader):
    """ Perform forward and backward pass on SIGN
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
    Args:
        model:      GAT model
        optimizer:  optimizer object
        loader:     train_loader containing data


    Returns:
        train_loss:     loss @ epoch
        train_f1:       f1 @ epoch
        delta_time:     from wrapper
    """
    model.train()

    total_examples = total_correct = total_loss = 0

    for batch in loader:
        batch_size = batch.batch_size

        # forward pass
        logits = model(
            batch.x.to(device),
            batch.edge_index.to(device)
        )[:batch_size]

        y = batch.y[:batch_size].to(logits.device)
        loss = F.nll_loss(logits, y)

        # store metrics
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_correct += int(sum(logits.argmax(dim=-1) == y))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        'f1': total_correct/total_examples,
        'loss': total_loss/total_examples,
    }


@torch.no_grad()
def test_epoch(model, data, subgraph_loader):
    """ Document validation or test loss and accuracy
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py

    Args:
        model:              GAT model
        data:               data
        subgraph_loader:    contain batch indices

    Returns:
        output:
            .inf_time
            .train_loss
            .val_loss
            .train_f1
            .val_f1
    """
    model.eval()

    logits, inf_time = model.inference(data.x, subgraph_loader)
    output = {'inf_time': inf_time}

    for split in ['train', 'val']:

        mask = eval(f'data.{split}_mask')
        mask_logits = logits[mask]
        mask_yhat = mask_logits.argmax(dim=-1)
        mask_y = data.y[mask].to(mask_logits.device)

        output.update({
            f'{split}_f1': (sum(mask_yhat == mask_y)/len(mask)).item(),
            f'{split}_loss': F.nll_loss(mask_logits, mask_y).item(),
        })

    return output


def main(args):
    set_seeds(args.seed)

    # data
    path = f'{args.dataset}_data.pth'
    data = download_data(path)
    data = standardize_data(data, args.dataset)

    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,  # can be bool or n_id indices
        num_neighbors=[args.n_neighbors]*args.nlayers,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,  # remove final batch if incomplete
        num_workers=args.num_workers,
    )

    subgraph_loader = NeighborLoader(
        copy.copy(data),
        input_nodes=None,
        num_neighbors=[-1]*args.nlayers,  # sample all neighbors
        shuffle=False,  # :batch_size in sequential order
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
    )

    subgraph_loader.data.num_nodes = data.num_nodes
    del subgraph_loader.data.x, subgraph_loader.data.y  # only need indices

    # model
    model = GAT(
        data.num_features,  # in_channel
        data.num_classes,  # out_channel
        args.hidden_channel,
        args.dropout,
        args.nlayers,
        args.heads_in,
        args.heads_out,
    ).to(device)

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

            eval_out = test_epoch(model, data, train_loader)

            scheduler.step(eval_out['val_loss'])

            # store epoch
            dict = {
                'run': run, 'epoch': epoch,
                'n_params': n_params, 'training_time': training_time
            }
            dict.update({f'training_{k}': v for k, v in training_out.items()})
            dict.update({f'eval_{k}': v for k, v in eval_out.items()})

            store_run = pd.concat(
                [store_run, pd.DataFrame(dict)],
                axis=0,
                ignore_index=True
            )

    return store_run.to_csv(
        f'{args.dataset}_GAT.csv',
        sep=',',
        header=True,
        index=False
    )


if __name__ == '__main__':
    main(args)
