import glob
import time
import argparse
import numpy as np
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ogb.nodeproppred import Evaluator

from transformation import transform_data

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
        gain = nn.init.calculate_gain('relu')
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


class SIGN(torch.nn.Module):
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
        super(SIGN, self).__init__()
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


def create_evaluator_fn(dataset: str):
    """ create function to determine accuracy score """

    if dataset in ['arxiv', 'products']:
        evaluator = Evaluator(name=f'ogbn-{dataset}')
        return lambda preds, labels: evaluator.eval({
            'y_true': labels.view(-1, 1),
            'y_pred': preds.view(-1, 1),
        })['acc']
    else:
        return lambda preds, labels: (preds == labels).numpy().mean()


def load_data(dataset):
    """ load original data (K=0) from dataset name """

    file_name = f'{dataset}_sign_k0.pth'
    path = glob.glob(f'./**/{file_name}', recursive=True)[0][2:]
    return torch.load(path)


@time_wrapper
def train(data, model, optimizer, train_loader):
    model.train()

    for batch in train_loader:
        xs = [data.x[batch].to(device)]
        xs += [data[f'x{i}'][batch].to(device)
               for i in range(1, model.K + 1)]
        loss = F.nll_loss(model(xs), data.y[batch].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval(data, model, eval_loader, evaluator):
    model.eval()

    @time_wrapper
    def inference(model, xs):
        return model(xs)

    inf_time = 0
    preds = []
    for batch in eval_loader:
        xs = [data.x[batch].to(device)]
        xs += [data[f'x{i}'][batch].to(device)
               for i in range(1, model.K + 1)]
        out, batch_time = inference(model, xs)

        # labels.append(data.y[batch].cpu())
        preds.append(out.argmax(dim=-1).cpu())
        inf_time += batch_time

    # labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    train_f1 = evaluator(preds[data.train_mask], data.y[data.train_mask])
    val_f1 = evaluator(preds[data.val_mask], data.y[data.val_mask])
    test_f1 = evaluator(preds[data.test_mask], data.y[data.test_mask])

    return train_f1, val_f1, test_f1, inf_time


def run(data, model, args, optimizer, train_loader, eval_loader, evaluator):
    """ execute single run to train/eval model"""
    model.reset_parameters()

    epoch_train_times = []
    epoch_inf_times = []
    best_epoch, best_val, best_test = 0, 0, 0

    for epoch in range(1, args.EPOCHS+1):
        _, train_time = train(data, model, optimizer, train_loader)
        epoch_train_times.append([train_time])

        # if (epoch == 1) or (epoch%5 == 0) or (epoch == args.EPOCHS):
        #     train_f1, val_f1, test_f1, inf_time = eval(
        #         data, model, eval_loader, evaluator)
        #     epoch_inf_times.append([inf_time])

        #     if val_f1 > best_val:
        #         best_epoch = epoch
        #         best_train, best_val, best_test = train_f1, val_f1, test_f1

        train_f1, val_f1, test_f1, inf_time = eval(
            data, model, eval_loader, evaluator)
        epoch_inf_times.append([inf_time])

        print('Epoch {}:, Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(
            epoch, train_f1, val_f1, test_f1))

        if val_f1 > best_val:
            best_epoch = epoch
            best_train, best_val, best_test = train_f1, val_f1, test_f1

    return {
        'best_epoch': [best_epoch],
        'best_train': [best_train],
        'best_val': [best_val],
        'best_test': [best_test],
        'train_times': [epoch_train_times],
        'inf_times': [epoch_inf_times],
    }


def main(args):
    # data
    data = load_data(args.DATASET)
    data, transform_time = transform_data(data, args)

    print(f'===== {args.DATASET} =====')

    print('Transformation Time (s): {:.4f}'.format(transform_time))

    train_loader = DataLoader(
        data.train_mask,
        batch_size=args.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    eval_loader = DataLoader(
        data.n_id,
        batch_size=args.EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    # model
    model = SIGN(
        data.num_features,       # in_channel
        data.num_classes,        # out_channel
        args.HIDDEN_CHANNEL,
        args.DROPOUT,
        args.INPUT_DROPOUT,
        args.K,
        args.N_FFLAYERS,
        args.BATCH_NORM
    ).to(device)

    print('# Model Params:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))

    # prep
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.LR,
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
        run_out = run(
            data,
            model,
            args,
            optimizer,
            train_loader,
            eval_loader,
            evaluator
        )

        print('Run {}: Best Epoch {}, Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(
            i, run_out['best_epoch'][0], run_out['best_train'][0],
            run_out['best_val'][0], run_out['best_test'][0]
        ))

        for k, v in run_out.items():
            results[k].append(v)  # store run results

    # print final numbers reported in thesis
    print('========================')
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
    print('========================')


if __name__ == '__main__':

    print(args)
    print()
    main(args)
