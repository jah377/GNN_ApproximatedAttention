
import copy

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import time_wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def gat_filter(data, args, GATdict=None):
    """ train GAT model and extract attention
    Args:
        data:
        GATtransform_params:     number of attention heads

    Returns:
        SparseTensor containing attention weights
    """

    # create neighbor samplers
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

    # build model
    model = GAT(
        data.num_features,  # in_channel
        data.num_classes,  # out_channel
        GATdict['hidden_channel'],
        GATdict['dropout'],
        GATdict['nlayers'],
        GATdict['heads_in'],
        GATdict['heads_out'],
    ).to(device)

    # build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GATdict['optimizer_lr'],
        weight_decay=GATdict['optimizer_decay'],
    )

    # build scheduler (modulates learning rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',     # nll_loss expected to decrease over epochs
        factor=0.1,     # lr reduction factor
        patience=5,     # reduce lr after _ epochs of no improvement
        min_lr=1e-6,    # min learning rate
        verbose=False,  # do not monitor lr updates
    )

    for _ in range(GATdict['epochs']):

        # train model
        model.train()
        for batch in train_loader:
            batch_size = batch.batch_size

            logits = model(
                batch.x.to(device),
                batch.edge_index.to(device)
            )[:batch_size]

            y = batch.y[:batch_size].to(logits.device)
            loss = F.nll_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # eval model
        model.eval()
        with torch.no_grad():
            logits, _ = model.inference(data.x, subgraph_loader)
            mask = data.val_mask
            val_loss = F.nll_loss(
                logits[mask].cpu(),
                data.y[mask].cpu()
            ).item()

        scheduler.step(val_loss)

    # extract attention matrix
    model.eval()
    with torch.no_grad():
        sparse_attn = model.extract_features(data.x, subgraph_loader)

    return sparse_attn
