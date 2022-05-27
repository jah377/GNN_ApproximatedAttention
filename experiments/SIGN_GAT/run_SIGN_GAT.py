import argparse
import pandas as pd
from distutils.util import strtobool

import torch
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_SIGN import SIGN
from transform_SamplerGAT import GATAttention
from steps_SIGN import train_epoch, test_epoch, get_inference_time
from general.utils import set_seeds, download_data, standardize_data, create_loader, get_n_params
from SamplerGAT_configs import GATparams

# product: https://arxiv.org/pdf/2004.11198v2.pdf
parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='products')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--optimizer_lr', type=float, default=0.0001)
parser.add_argument('--optimizer_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--hidden_channel', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--batch_norm', type=strtobool, default=True)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--return_results', type=strtobool, default=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    assert args.dataset in GATparams.keys()
    set_seeds(args.seed)

    # data
    data = download_data(args.dataset, K=args.K)
    data = standardize_data(data, args.dataset)
    data, transform_time = GATAttention(
        data, args.K, GATparams.get(args.dataset))
    print('-- TRANSFORM COMPLETE ')

    train_loader = create_loader(data, 'train', batch_size=args.batch_size)
    val_loader = create_loader(data, 'val', batch_size=args.batch_size)
    test_loader = create_loader(data, 'test', batch_size=args.batch_size)
    all_loader = create_loader(data, 'all', batch_size=args.batch_size)

    # model
    model = SIGN(
        data.num_features,       # in_channel
        data.num_classes,        # out_channel
        args.hidden_channel,
        args.dropout,
        args.K,
        args.batch_norm).to(device)

    n_params = get_n_params(model)

    # evaluator, if ogbn dataset
    if data.dataset_name in ['products', 'arxiv']:
        evaluator = Evaluator(name=f'ogbn-{data.dataset_name}')
    else:
        evaluator = None

    store_run = pd.DataFrame()
    store_inf_time = []
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
                model, data, optimizer, train_loader)

            train_out = test_epoch(model, data, train_loader, evaluator)
            val_out = test_epoch(model, data, val_loader, evaluator)
            test_out = test_epoch(model, data, test_loader, evaluator)

            scheduler.step(val_out['loss'])

            # store epoch
            epoch_dict = {
                'run': run,
                'epoch': epoch,
                'n_params': n_params,
                'training_time': training_time
            }
            epoch_dict.update(
                {f'training_{k}': v for k, v in training_out.items()})
            epoch_dict.update(
                {f'eval_train_{k}': v for k, v in train_out.items()})
            epoch_dict.update({f'eval_val_{k}': v for k, v in val_out.items()})
            epoch_dict.update(
                {f'eval_test_{k}': v for k, v in test_out.items()})

            store_run = pd.concat(
                [store_run, pd.DataFrame.from_dict([epoch_dict])],
                ignore_index=True
            )

        # total inference time
        store_inf_time.append(
            get_inference_time(model, data, all_loader)
        )

    if args.return_results:

        print(f' --- {data.dataset_name.upper()} --- ')

        # parameters
        n_params = store_run['n_params'].mean()
        print(f'Number of Model Parameters: {n_params}')

        # transformation time
        print(f'Precomputation Time (s): {transform_time}')

        # training time
        train_times = store_run['training_time'].agg(['mean', 'std'])
        print(
            f'Training Time (s): {train_times[0].round(5)} +/- {train_times[1].round(5)}')

        import numpy as np
        inf_mean, inf_std = np.mean(store_inf_time), np.std(store_inf_time)
        print(
            f'Inference Time (s): {inf_mean.__round__(5)} +/- {inf_std.__round__(5)}')

        # f1 score
        last_epoch = (store_run.epoch == max(store_run.epoch))
        f1_scores = store_run[last_epoch]['eval_test_f1'].agg(['mean', 'std'])
        print(
            f'F1 Score (s): {f1_scores[0].round(5)} +/- {f1_scores[1].round(5)}')

        print(f' ---------------- ')


if __name__ == '__main__':
    main(args)
