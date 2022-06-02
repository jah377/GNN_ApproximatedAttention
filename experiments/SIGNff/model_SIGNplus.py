import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """
    https://github.com/THUDM/CRGNN/blob/main/layer.py
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/sign/sign.py
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        hidden_channel: int,
        dropout: float,
        n_fflayers: int,
        batch_norm: bool = True
    ):

        super(FeedForwardNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel
        self.dropout = dropout
        self.n_fflayers = max(1, n_fflayers)
        self.batch_norm = batch_norm
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if n_fflayers == 1:
            self.lins.append(nn.Linear(in_channel, out_channel))
        else:
            self.lins.append(nn.Linear(in_channel, hidden_channel))
            self.bns.append(nn.BatchNorm1d(hidden_channel))

            for _ in range(n_fflayers-2):
                self.lins.append(nn.Linear(hidden_channel, hidden_channel))
                self.bns.append(nn.BatchNorm1d(hidden_channel))

            self.lins.append(nn.Linear(hidden_channel, out_channel))

        if self.n_fflayers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            nn.init.zeros_(lin.bias)
        for bns in self.bns:
            bns.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.lins):
            x = layer(x)
            if i < self.n_fflayers-1:
                if self.batch_norm == True:
                    x = self.dropout(self.prelu(self.bns[i](x)))
                else:
                    x = self.dropout(self.prelu(x))
        return x


class SIGN_plus(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        hidden_channel: int,
        dropout: float,
        input_dropout: float,
        K: int,
        n_fflayers: int,
        batch_norm: bool = True
    ):
        super(SIGN_plus, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel
        self.K = K
        self.n_fflayers = n_fflayers
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(input_dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()

        # inception feedforward layers
        for _ in range(self.K + 1):
            self.inception_ffs.append(
                FeedForwardNet(
                    in_channel, hidden_channel, hidden_channel,
                    dropout, n_fflayers, batch_norm
                )
            )

        # feedforward layer for concatenated outputs
        self.concat_ff = FeedForwardNet(
            (self.K+1)*hidden_channel, out_channel,
            hidden_channel, dropout, n_fflayers, batch_norm
        )

    def reset_parameters(self):
        for layer in self.inception_ffs:
            layer.reset_parameters()
        self.concat_ff.reset_parameters()

    def forward(self, xs):
        """ xs = [AX^0, AX^1, ..., AX^K] """

        xs = [self.input_dropout(x) for x in xs]  # input dropout
        hs = []  # store forward pass of each AX^K

        for i, layer in enumerate(self.inception_ffs):
            hs.append(layer(xs[i]))

        return self.concat_ff(self.dropout(self.prelu(torch.cat(hs, dim=-1)))).log_softmax(dim=-1)
