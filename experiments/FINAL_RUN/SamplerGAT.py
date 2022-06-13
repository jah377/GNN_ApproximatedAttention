import glob
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from distutils.util import strtobool

from utils import time_wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='SIGN')
parser.add_argument('--SEED', type=int, default=42)
parser.add_argument('--EPOCHS', type=int, default=1000)
parser.add_argument('--HIDDEN_CHANNEL', type=int, default=512)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--LR', type=float, default=0.001)
parser.add_argument('--DATASET', type=str, default='arxiv')
parser.add_argument('--DROPOUT', type=float, default=0.5)
parser.add_argument('--WEIGHT_DECAY', type=float, default=0)
parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=50000)
parser.add_argument('--EVAL_BATCH_SIZE', type=int, default=100000)
parser.add_argument('--N_FFLAYERS', type=int, default=2)
parser.add_argument('--INPUT_DROPOUT', type=float, default=0)
parser.add_argument('--BATCH_NORM', type=strtobool, default=True)
parser.add_argument('--TRANSFORMATION', type=str, default='sign')
parser.add_argument('--CS_BATCH_SIZE', type=int, default=10000)
parser.add_argument('--ATTN_HEADS', type=float, default=2)
parser.add_argument('--DPA_NORM', type=strtobool, default=True)
parser.add_argument('--N_RUNS', type=int, default=10)
args = parser.parse_args()


class GAT(torch.nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 hidden_channel,
                 dropout,
                 nlayers,
                 heads_in,
                 heads_out):
        """
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
                x = x_all[batch.n_id.to(x_all.device)].to(device)
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
                x = x_all[batch.n_id.to(x_all.device)].to(device)

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
                    values = attn_w.mean(dim=1).detach().cpu()
                    attn_coo += torch.sparse_coo_tensor(
                        batch.n_id[attn_i].cpu(),
                        attn_w.mean(dim=1).detach().cpu(),
                        size=(dim, dim)
                    )

                    count_total += torch.sparse_coo_tensor(
                        batch.n_id[attn_i].cpu(),
                        torch.ones_like(values),
                        size=(dim, dim)
                    )
                    del values, attn_i, attn_w

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


def train(model, optimizer, train_loader):
    model.train()
    total_nodes = total_loss = 0

    for batch in train_loader:
        batch_size = batch.batch_size
        logits = model(
            batch.x.to(device),
            batch.edge_index.to(device)
        )[:batch_size]
        loss = F.nll_loss(logits, batch.y[:batch_size].to(logits.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_nodes += batch_size
        total_loss += float(loss) * batch_size

    return float(total_loss/total_nodes)


def eval(model, data, subgraph_loader):
    model.eval()

    logits, inf_time = model.inference(data.x, subgraph_loader)

    train_f1 = (logits[data.train_mask].argmax(dim=-1) ==
                data.y[data.train_mask]).numpy().mean()
    val_f1 = (logits[data.val_mask].argmax(dim=-1) ==
              data.y[data.val_mask]).numpy().mean()
    test_f1 = (logits[data.test_mask].argmax(dim=-1) ==
               data.y[data.test_mask]).numpy().mean()
    val_loss = float(F.nll_loss(
        logits[data.val_mask].cpu(), data.y[data.val_mask].cpu()))

    return train_f1, val_f1, test_f1, val_loss, inf_time
