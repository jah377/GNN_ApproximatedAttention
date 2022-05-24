import torch
import argparse
import pandas as pd
from distutils.util import strtobool
from torch.optim.lr_scheduler import ReduceLROnPlateau

from general.models.SIGN import net as SIGN
from general.utils import set_seeds, download_data, standardize_data, create_loader
from general.epoch_steps.steps_SIGN import train_epoch, test_epoch
from general.transforms.transforms_DotProduct import transform_wAttention


#################################################################
########## THIS SHOULD BE IDENTICAL TO HPS_SIGN_SHA.PY ##########
#################################################################

# product: https://arxiv.org/pdf/2004.11198v2.pdf
parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='products')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--optimizer_lr', type=float, default=0.0001)
parser.add_argument('--optimizer_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--hidden_channel', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--batch_norm', type=strtobool, default=True)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--attn_heads', type=int,
                    default=1)  # hyperparameter for MHA
parser.add_argument('--return_results', type=strtobool, default=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    set_seeds(args.seed)

    # data
    data = download_data(args.dataset, K=args.K)
    data = standardize_data(data, args.dataset)
    data, transform_time = transform_wAttention(data, args.K, args.attn_heads)
    print('-- TRANSFORM COMPLETE ')

    train_loader = create_loader(data, 'train', batch_size=args.batch_size)
    val_loader = create_loader(data, 'val', batch_size=args.batch_size)
    test_loader = create_loader(data, 'test', batch_size=args.batch_size)

    # model
    model = SIGN(
        data.num_features,       # in_channel
        data.num_classes,        # out_channel
        args.hidden_channel,
        args.dropout,
        args.K,
        args.batch_norm).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    store_run = pd.DataFrame()
    for run in range(args.n_runs):
        model.reset_parameters

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

            train_out = test_epoch(model, data, train_loader)
            val_out = test_epoch(model, data, val_loader)
            test_out = test_epoch(model, data, test_loader)

            scheduler.step(val_out['loss'])

            # store epoch
            epoch_dict = {
                'run': run, 'transform_time': transform_time, 'epoch': epoch,
                'n_params': n_params, 'training_time': training_time
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

    if args.return_results:
        return store_run.to_csv(
            f'{args.dataset}_output.csv',
            sep=',',
            header=True,
            index=False
        )

    return '-- RUN COMPLETE '


if __name__ == '__main__':
    main(args)
