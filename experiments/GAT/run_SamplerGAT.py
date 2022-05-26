import copy
import argparse
import pandas as pd
from distutils.util import strtobool

import torch
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import NeighborLoader

from model_SamplerGAT import GAT
from steps_SamplerGAT import train_epoch, test_epoch
from general.utils import set_seeds, download_data, standardize_data

# product: https://arxiv.org/pdf/2004.11198v2.pdf
parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='products')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--optimizer_lr', type=float, default=0.0001)
parser.add_argument('--optimizer_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--hidden_channel', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--heads_in', type=int, default=8)
parser.add_argument('--heads_out', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1789)
parser.add_argument('--n_neighbors', type=int, default=150)
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--return_results', type=strtobool, default=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    print(args)
    set_seeds(args.seed)

    # data
    data = download_data(args.dataset)
    data = standardize_data(data, args.dataset)

    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,  # can be bool or n_id indices
        num_neighbors=[args.n_neighbors]*args.nlayers,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,  # remove final batch if incomplete
    )

    subgraph_loader = NeighborLoader(
        copy.copy(data),
        input_nodes=None,
        num_neighbors=[-1]*args.nlayers,  # sample all neighbors
        shuffle=False,                          # :batch_size in sequential order
        batch_size=args.batch_size,
        drop_last=False,
    )
    subgraph_loader.data.num_nodes = data.num_nodes
    del subgraph_loader.data.x, subgraph_loader.data.y  # only need indices

    # model
    model = GAT(
        data.num_features,       # in_channel
        data.num_classes,        # out_channel
        args.hidden_channel,
        args.dropout,
        args.nlayers,
        args.heads_in,
        args.heads_out,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # evaluator, if ogbn dataset
    if data.dataset_name in ['products', 'arxiv']:
        evaluator = Evaluator(name=f'ogbn-{data.dataset_name}')
    else:
        evaluator = None

    store_run = pd.DataFrame()
    for run in range(args.n_runs):
        model.reset_parameters()

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
                model, optimizer, train_loader)

            test_out, inf_time = test_epoch(
                model, data, subgraph_loader, evaluator)

            val_loss = test_out['val_loss']
            scheduler.step(val_loss)

            # store epoch
            epoch_dict = {
                'run': run, 'epoch': epoch,
                'n_params': n_params, 'training_time': training_time, 'inf_time': inf_time,
            }
            epoch_dict.update(
                {f'training_{k}': v for k, v in training_out.items()})
            epoch_dict.update(
                {f'eval_{k}': v for k, v in test_out.items()})

            store_run = pd.concat(
                [store_run, pd.DataFrame.from_dict([epoch_dict])],
                ignore_index=True
            )

    if args.return_results:

        print(f' --- {data.dataset_name.upper()} --- ')

        # parameters
        n_params = store_run['n_params'].mean()
        print(f'Number of Model Parameters: {n_params}')

        # total train & inference times
        train_times = store_run['training_time'].agg(['mean', 'std'])
        print(
            f'Training Time (s): {train_times[0].round(5)} +/- {train_times[1].round(5)}')

        inf_time = store_run['inf_time'].agg(['mean', 'std'])
        print(
            f'Inference Time (s): {inf_time[0].round(5)} +/- {inf_time[1].round(5)}')

        # f1 score
        last_epoch = (store_run.epoch == max(store_run.epoch))
        f1_scores = store_run[last_epoch]['eval_test_f1'].agg(['mean', 'std'])
        print(
            f'F1 Score (s): {f1_scores[0].round(5)} +/- {f1_scores[1].round(5)}')

        print(f' ---------------- ')


if __name__ == '__main__':
    main(args)
