import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.lin.reset_parameters()

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
