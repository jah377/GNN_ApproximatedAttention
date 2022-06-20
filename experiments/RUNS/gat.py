import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader

import pytorch_model_summary as pms

from utils import set_seeds, time_wrapper, load_data, create_evaluator_fn


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


@time_wrapper
def train(model, optimizer, train_loader):
    model.train()

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


@torch.no_grad()
def eval(model, data, subgraph_loader, evaluator):
    model.eval()

    out, inf_time = model.inference(data.x, subgraph_loader)
    preds = out.argmax(dim=-1).cpu()

    train_f1 = evaluator(preds[data.train_mask],
                         data.y[data.train_mask].to(preds.device))
    val_f1 = evaluator(preds[data.val_mask],
                       data.y[data.val_mask].to(preds.device))
    test_f1 = evaluator(preds[data.test_mask],
                        data.y[data.test_mask].to(preds.device))

    return train_f1, val_f1, test_f1, inf_time


def run(data, model, args, optimizer, train_loader, subgraph_loader, evaluator):
    """ execute single run to train/eval model"""
    model.reset_parameters()

    epoch_train_times = []
    epoch_inf_times = []
    best_epoch, best_val, best_test = 0, 0, 0

    for epoch in range(1, args.EPOCHS+1):
        _, train_time = train(model, optimizer, train_loader)
        epoch_train_times.append([train_time])

        if (epoch % args.EVAL_EVERY == 0) or (epoch == args.EPOCHS):
            train_f1, val_f1, test_f1, inf_time = eval(
                model, data, subgraph_loader, evaluator)
            epoch_inf_times.append([inf_time])

            if val_f1 > best_val:
                best_epoch = epoch
                best_train, best_val, best_test = train_f1, val_f1, test_f1

            print('Epoch {}:, Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(
                epoch, train_f1, val_f1, test_f1))

    return {
        'best_epoch': [best_epoch],
        'best_train': [best_train],
        'best_val': [best_val],
        'best_test': [best_test],
        'train_times': [epoch_train_times],
        'inf_times': [epoch_inf_times],
    }


def main(args):
    set_seeds(args.SEED)

    # load data
    data = load_data(args.DATASET)

    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[args.NEIGHBORS]*args.LAYERS,
        shuffle=True,
        batch_size=args.BATCH_SIZE,
        drop_last=True,
    )

    subgraph_loader = NeighborLoader(
        copy.copy(data),
        input_nodes=None,
        num_neighbors=[-1]*args.LAYERS,
        shuffle=False,
        batch_size=args.BATCH_SIZE,
        drop_last=False,
    )
    subgraph_loader.data.num_nodes = data.num_nodes
    del subgraph_loader.data.x, subgraph_loader.data.y  # keep indices

    # build model
    model = GAT(
        data.num_features,  # in_channel
        data.num_classes,   # out_channel
        args.HIDDEN_UNITS,
        args.NODE_DROPOUT,
        args.LAYERS,
        args.HEADS_IN,
        args.HEADS_OUT,
    ).to(device)

    print('# Model Params:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    print()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.LEARNING_RATE,
        weight_decay=args.WEIGHT_DECAY
    )

    evaluator = create_evaluator_fn(args.DATASET)

    # per run
    results = {
        'best_epoch': [], 'best_train': [],
        'best_val': [], 'best_test': [],
        'train_times': [], 'inf_times': [],
    }

    for i in range(1, args.N_RUNS+1):
        print(f'RUN #{i}:')
        run_out = run(
            data,
            model,
            args,
            optimizer,
            train_loader,
            subgraph_loader,
            evaluator
        )

        print('Run {}: Best Epoch {}, Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(
            i, run_out['best_epoch'][0], run_out['best_train'][0],
            run_out['best_val'][0], run_out['best_test'][0]
        ))
        print()

        for k, v in run_out.items():
            results[k].append(v)  # store run results

    # print final numbers reported in thesis
    print('\n\n')
    print('==================================================')
    print('Avg. Training Time (epoch) (s): {:.4f} +/- {:.4f}'.format(
        np.mean(results['train_times']), np.std(results['train_times'])))
    print('Avg. Inference Time (s): {:.4f} +/- {:.4f}'.format(
        np.mean(results['inf_times']), np.std(results['inf_times'])))

    print('Avg. Training Acc: {:.4f} +/- {:.4f}'.format(
        np.mean(results['best_train']), np.std(results['best_train'])))
    print('Avg. Validation Acc: {:.4f} +/- {:.4f}'.format(
        np.mean(results['best_val']), np.std(results['best_val'])))
    print('Avg. Test Acc: {:.4f} +/- {:.4f}'.format(
        np.mean(results['best_test']), np.std(results['best_test'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inputs')
    parser.add_argument('--DATASET', type=str, default='cora',
                        help='name of dataset')
    parser.add_argument('--EPOCHS', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--EVAL_EVERY', type=int, default=5,
                        help='evaluate model every _ epochs')
    parser.add_argument('--N_RUNS', type=int, default=2,
                        help='number of runs')
    parser.add_argument('--SEED', type=int, default=42, help='seed value')
    parser.add_argument('--BATCH_SIZE', type=int, default=2048,
                        help='DataLoader batch size')
    parser.add_argument('--LEARNING_RATE', type=float,
                        default=0.01, help='optimizer learning rate')
    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.01,
                        help='optimizer regularization param')
    parser.add_argument('--NODE_DROPOUT', type=float, default=0.3,
                        help='fraction of NN nodes to be dropped')
    parser.add_argument('--HIDDEN_UNITS', type=int,
                        default=8, help='hidden channel size')
    parser.add_argument('--LAYERS', type=int, default=2,
                        help='number of GAT layers')
    parser.add_argument('--HEADS_IN', type=int, default=8,
                        help='number of attn heads used for all but final GAT layer')
    parser.add_argument('--HEADS_OUT', type=int, default=8,
                        help='number of attn heads in final GAT layer')
    parser.add_argument('--NEIGHBORS', type=int, default=150,
                        help='number of neighbors sampled per node, per layer')
    args = parser.parse_args()

    print()
    print(f'===== {args.DATASET.upper()} =====')
    print()
    print(args)
    print()
    main(args)
