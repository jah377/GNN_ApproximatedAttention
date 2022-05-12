import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from general.utils import resources

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class net(torch.nn.Module):
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
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_gat.py

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

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    @resources
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        for i, conv in enumerate(self.convs):
            xs = []

            for batch in subgraph_loader:

                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))

                # activation for all but final GATConv
                if i < len(self.convs) - 1:
                    x = F.relu(x)

                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)

        return F.log_softmax(x_all, dim=-1)

    @torch.no_grad()
    def extract_features(self, x_all, subgraph_loader):
        """
        single edge_index may be found in multiple batches
        - average across batches
        """
        cpu = torch.device('cpu')

        # to store edge_index attention weights and occurances
        dim = subgraph_loader.data.num_nodes
        attn_coo = torch.sparse_coo_tensor(size=(dim, dim)).to(cpu)
        count_total = torch.sparse_coo_tensor(size=(dim, dim)).to(cpu)

        for i, conv in enumerate(self.convs):
            xs = []

            for batch in subgraph_loader:

                x = x_all[batch.n_id.to(x_all.device)].to(device)

                if i < len(self.convs)-1:
                    x = conv(x, batch.edge_index.to(device))
                    x = F.relu(x)
                else:
                    # extract attention weights on final GATConv layer
                    x, (attn_i, attn_w) = conv(
                        x,
                        batch.edge_index.to(device),
                        return_attention_weights=True
                    )

                    # accumulate weights and number of edge_index occurances
                    indices = batch.n_id[attn_i].to(cpu)
                    values = attn_w.mean(dim=1).detach().to(cpu)

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

        r, c = attn_coo.indices()

        return SparseTensor(
            row=r,
            col=c,
            value=attn_coo.values().detach(),
            sparse_sizes=(dim, dim)
        )  # to replace adj_t
