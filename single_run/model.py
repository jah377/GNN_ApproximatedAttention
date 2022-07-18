import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeedForwardNet(nn.Module):
    """
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/sign/sign.py
    """

    def __init__(
        self,
        in_units: int,              # input units
        out_units: int,             # output units
        hidden_channels: int,       # hidden units
        node_dropout: float,        # nn regularization
        n_layers: int,              # nn layers
        batch_norm: bool = True     # include batch normalization
    ):
        super(FeedForwardNet, self).__init__()
        self.n_layers = max(1, n_layers)
        self.batch_norm = batch_norm
        self.prelu = nn.PReLU()
        self.node_dropout = nn.Dropout(node_dropout)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if self.n_layers == 1:
            self.lins.append(nn.Linear(in_units, out_units))
        else:
            self.lins.append(nn.Linear(in_units, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            for _ in range(self.n_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            self.lins.append(nn.Linear(hidden_channels, out_units))

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            nn.init.zeros_(lin.bias)
        for bns in self.bns:
            bns.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.lins):
            x = layer(x)
            if i < self.n_layers-1:
                if self.batch_norm == True:
                    x = self.node_dropout(self.prelu(self.bns[i](x)))
                else:
                    x = self.node_dropout(self.prelu(x))
        return x


class SIGN(torch.nn.Module):
    def __init__(
        self,
        in_units: int,                  # feature length
        out_units: int,                 # num classes
        inception_units: int,           # hidden units
        inception_layers: int,          # nn module layers
        classification_units: int,      # hidden units
        classification_layers: int,     # nn module layers
        feature_dropout: float,         # input dropout
        node_dropout: float,            # nn regularization
        hops: int,                      # K-hop aggregation
        batch_norm: bool = True         # include batch normalization
    ):
        super(SIGN, self).__init__()
        self.hops = hops
        self.inception_layers = inception_layers
        self.classification_layers = classification_layers
        self.batch_norm = batch_norm
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.node_dropout = nn.Dropout(node_dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()

        # inception feedforward layers
        for _ in range(self.hops + 1):
            self.inception_ffs.append(
                FeedForwardNet(
                    in_units,
                    inception_units,
                    inception_units,
                    node_dropout,
                    inception_layers,
                    batch_norm
                )
            )

        # feedforward layer for concatenated outputs
        self.concat_ff = FeedForwardNet(
            (self.hops+1)*inception_units,
            out_units,
            classification_units,
            node_dropout,
            classification_layers,
            batch_norm
        )

    def reset_parameters(self):
        for layer in self.inception_ffs:
            layer.reset_parameters()
        self.concat_ff.reset_parameters()

    def forward(self, xs):
        """ xs = [AX, A(AX), ..., AX^K] """

        xs = [self.feature_dropout(x) for x in xs]
        hs = []

        for hop, layer in enumerate(self.inception_ffs):
            hs.append(layer(xs[hop]))

        return self.concat_ff(
            self.node_dropout(
                self.prelu(
                    torch.cat(hs, dim=-1)
                ))).log_softmax(dim=-1)
