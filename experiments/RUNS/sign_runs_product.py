import argparse
import numpy as np
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor
import pytorch_model_summary as pms

from transform_cs import cosine_filter
from transform_dp import dotproduct_filter
from transform_gat import gat_filter
from utils import set_seeds, time_wrapper, load_data, create_loader, create_evaluator_fn, print_filter_stats


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeedForwardNet(nn.Module):
    """
    https://github.com/THUDM/CRGNN/blob/main/layer.py
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/sign/sign.py
    """

    def __init__(
        self,
        in_units: int,
        out_units: int,
        hidden_channels: int,
        node_dropout: float,
        n_layers: int,
        batch_norm: bool = True
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
        inception_layers: int,          # n layers
        classification_units: int,      # hidden units
        classification_layers: int,     # n layers
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


@time_wrapper
def train(data, model, optimizer, train_loader):
    model.train()

    for batch in train_loader:
        xs = [data.x[batch].to(device)]
        xs += [data[f'x{i}'][batch].to(device)
               for i in range(1, model.hops + 1)]
        out = model(xs)
        loss = F.nll_loss(out, data.y[batch].to(out.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@time_wrapper
def inference(model, xs):
    """ return logits and inference time """
    return model(xs)


@torch.no_grad()
def eval(data, model, eval_loader, evaluator):
    model.eval()

    inf_time = 0
    preds, labels = [], []
    for batch in eval_loader:
        xs = [data.x[batch].to(device)]
        xs += [data[f'x{i}'][batch].to(device)
               for i in range(1, model.hops + 1)]
        out, batch_time = inference(model, xs)

        labels.append(data.y[batch].cpu())
        preds.append(out.argmax(dim=-1).cpu())
        inf_time += batch_time

    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    train_f1 = evaluator(preds[data.train_mask], data.y[data.train_mask])
    val_f1 = evaluator(preds[data.val_mask], data.y[data.val_mask])
    test_f1 = evaluator(preds[data.test_mask], data.y[data.test_mask])

    return train_f1, val_f1, test_f1, inf_time


@time_wrapper
def transform_data(data, args):
    """ SIGN transformation with or without attention filter """
    filter = 'sign' if args.ATTN_FILTER == None else args.ATTN_FILTER.lower()
    has_corr_trans = filter in ['sign', 'gat',
                                'cosine', 'dot_product', 'cosine_per_k']
    assert has_corr_trans, "Transformation: Must enter 'none','gat', 'cosine', 'dot_product', or 'cosine_per_k'"
    assert data.x is not None,  f'Dataset: data.x empty'

    xs = [data.x]  # store transformations

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

    # replace adj_t with attention filter
    if filter == 'cosine':
        adj_t = cosine_filter(data.x, data.edge_index, args)
    elif filter == 'gat':
        adj_t = gat_filter(data, args)
    elif filter == 'dot_product':
        adj_t = dotproduct_filter(data, args)
    elif filter == 'cosine_per_k':
        for i in range(1, args.HOPS + 1):
            adj_t = cosine_filter(xs[-1], data.edge_index, args)
            print_filter_stats(adj_t)
            adj_t = deg_inv_sqrt.view(-1, 1) * \
                adj_t * deg_inv_sqrt.view(1, -1)
            xs += [adj_t @ xs[-1]]
            data[f'x{i}'] = xs[-1]
            del adj_t
        return data

    print(filter.upper())
    print_filter_stats(adj_t)
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    for i in range(1, args.HOPS + 1):
        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    return data


def main(args):
    results = {
        'best_epoch': [], 'best_train': [],
        'best_val': [], 'best_test': [],
        'preproc_time': [], 'train_times': [], 'inf_times': [],
    }

    data = load_data(args.DATASET)
    data, transform_time = transform_data(data, args)
    print('Total Transformation Time: {:0.4f}'.format(transform_time))
    train_loader = create_loader(data, 'train', args.BATCH_SIZE)
    eval_loader = create_loader(data, 'all', args.BATCH_SIZE)

    for i, seed in enumerate(args.RUN_SEEDS):
        try:
            model.reset_parameters()
        except:
            pass

        print(f'RUN #{i}: seed={seed}')
        set_seeds(seed)

        # data = load_data(args.DATASET)
        # data, transform_time = transform_data(data, args)
        # print('Total Transformation Time: {:0.4f}'.format(transform_time))
        # train_loader = create_loader(data, 'train', args.BATCH_SIZE)
        # eval_loader = create_loader(data, 'all', args.BATCH_SIZE)

        model = SIGN(
            data.num_features,
            data.num_classes,
            args.INCEPTION_UNITS,
            args.INCEPTION_LAYERS,
            args.CLASSIFICATION_UNITS,
            args.CLASSIFICATION_LAYERS,
            args.FEATURE_DROPOUT,
            args.NODE_DROPOUT,
            args.HOPS,
            args.BATCH_NORMALIZATION,
        ).to(device)

        model_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.LEARNING_RATE,
            weight_decay=args.WEIGHT_DECAY
        )

        evaluator = create_evaluator_fn(args.DATASET)

        # train and evaluate
        epoch_train_times = []
        epoch_inf_times = []
        best_epoch, best_val, best_test = 0, 0, 0

        for epoch in range(1, args.EPOCHS+1):
            _, train_time = train(data, model, optimizer, train_loader)
            epoch_train_times.append([train_time])

            if (epoch % args.EVAL_EVERY == 0) or (epoch == args.EPOCHS):
                train_f1, val_f1, test_f1, inf_time = eval(
                    data, model, eval_loader, evaluator)
                epoch_inf_times.append([inf_time])

                if val_f1 > best_val:
                    best_epoch = epoch
                    best_train, best_val, best_test = train_f1, val_f1, test_f1

                print('Epoch {}:, Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(
                    epoch, train_f1, val_f1, test_f1))

        # print store best values of run
        run_results = {
            'preproc_time': [transform_time],
            'best_epoch': [best_epoch],
            'best_train': [best_train],
            'best_val': [best_val],
            'best_test': [best_test],
            'train_times': [epoch_train_times],
            'inf_times': [epoch_inf_times],
        }

        for k, v in run_results.items():
            results[k].append(v)

        print('BEST: Epoch {}, Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(
            best_epoch, best_train, best_val, best_test))
        print()

    # print final numbers reported in thesis
    print('\n\n')
    print('==================================================')
    print('Model Parameters: {}'.format(model_params))
    print()
    print('Avg. Preaggregation Time (s): {:.4f} +/- {:.4f}'.format(
        np.mean(results['preproc_time']), np.std(results['preproc_time'])))
    print('Avg. Training Time (epoch) (s): {:.4f} +/- {:.4f}'.format(
        np.mean(results['train_times']), np.std(results['train_times'])))
    print('Avg. Inference Time (s): {:.4f} +/- {:.4f}'.format(
        np.mean(results['inf_times']), np.std(results['inf_times'])))
    print()
    print('Avg. Training Acc: {:.4f} +/- {:.4f}'.format(
        np.mean(results['best_train']), np.std(results['best_train'])))
    print('Avg. Validation Acc: {:.4f} +/- {:.4f}'.format(
        np.mean(results['best_val']), np.std(results['best_val'])))
    print('Avg. Test Acc: {:.4f} +/- {:.4f}'.format(
        np.mean(results['best_test']), np.std(results['best_test'])))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inputs')
    parser.add_argument('--DATASET', type=str, default='cora',
                        help='name of dataset')
    parser.add_argument('--ATTN_FILTER', type=str, default=None,
                        help='None, gat, cosine, dot_product, or cosine_per_k')
    parser.add_argument('--EPOCHS', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--EVAL_EVERY', type=int, default=5,
                        help='evaluate model every _ epochs')
    parser.add_argument('--RUN_SEEDS', '--list', help='delimited list input of run seeds',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--HOPS', type=int, default=2,
                        help='k-hop neighborhood aggregations')
    parser.add_argument('--BATCH_SIZE', type=int, default=2048,
                        help='DataLoader batch size')
    parser.add_argument('--LEARNING_RATE', type=float,
                        default=0.01, help='optimizer learning rate')
    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.01,
                        help='optimizer regularization param')
    parser.add_argument('--INCEPTION_LAYERS', type=int, default=1,
                        help='number of inception feed-forward layers')
    parser.add_argument('--INCEPTION_UNITS', type=int,
                        default=512, help='inception hidden channel size')
    parser.add_argument('--CLASSIFICATION_LAYERS', type=int, default=1,
                        help='number of classification feed-forward layers ')
    parser.add_argument('--CLASSIFICATION_UNITS', type=int,
                        default=512, help='classification hidden channel size')
    parser.add_argument('--FEATURE_DROPOUT', type=float,
                        default=0.4, help='fraction of features to be dropped')
    parser.add_argument('--NODE_DROPOUT', type=float, default=0.3,
                        help='fraction of NN nodes to be dropped')
    parser.add_argument('--BATCH_NORMALIZATION', type=strtobool,
                        default=True, help='NN regularization')
    parser.add_argument('--ATTN_HEADS', type=int, default=2,
                        help='number of attention heads (DPA only)')
    parser.add_argument('--ATTN_NORMALIZATION', type=strtobool, default=True,
                        help='row min-max normalization of DPA/CS attn weights')
    parser.add_argument('--GAT_EPOCHS', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--GAT_BATCH_SIZE', type=int, default=1024,
                        help='DataLoader batch size')
    parser.add_argument('--GAT_LEARNING_RATE', type=float,
                        default=0.01, help='optimizer learning rate')
    parser.add_argument('--GAT_WEIGHT_DECAY', type=float, default=0.001,
                        help='optimizer regularization param')
    parser.add_argument('--GAT_HIDDEN_UNITS', type=int,
                        default=8, help='hidden channel size')
    parser.add_argument('--GAT_NODE_DROPOUT', type=float, default=0.6,
                        help='fraction of NN nodes to be dropped')
    parser.add_argument('--GAT_LAYERS', type=int, default=2,
                        help='number of GAT layers')
    parser.add_argument('--GAT_HEADS_IN', type=int, default=8,
                        help='number of attn heads used for all but final GAT layer')
    parser.add_argument('--GAT_HEADS_OUT', type=int, default=8,
                        help='number of attn heads in final GAT layer')
    parser.add_argument('--GAT_NEIGHBORS', type=int, default=150,
                        help='number of neighbors sampled per node, per layer')
    parser.add_argument('--GAT_LR_PATIENCE', type=int, default=5,
                        help='scheduler updates LR after __ epochs')
    args = parser.parse_args()

    print()
    print(f'===== {args.DATASET.upper()} =====')
    print()
    print(args)
    print()
    main(args)
