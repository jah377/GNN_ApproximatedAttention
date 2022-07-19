import os
import os.path as osp

import time
import argparse
import numpy as np
from distutils.util import strtobool

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from model import SIGN
from attn_filters.filter_cosine import cosine_filter
from attn_filters.filter_dotprod import dotproduct_filter
from attn_filters.filter_gat import gat_filter
from utils import set_seeds, prep_data, time_wrapper, create_loader, create_evaluator_fn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    filter_time = 0  # for baseline SIGN
    if filter == 'cosine':
        adj_t, filter_time = cosine_filter(data.x, data.edge_index, args)
    elif filter == 'gat':
        adj_t, filter_time = gat_filter(data, args)
    elif filter == 'dot_product':
        adj_t, filter_time = dotproduct_filter(data, args)

    # transform data
    start = time.time()
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    for i in range(1, args.HOPS + 1):
        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]
    diffusion_time = time.time()-start

    return data, filter_time, diffusion_time


def runs(args):
    results = {
        'best_epoch': [], 'best_train': [],
        'best_val': [], 'best_test': [],
        'filter_time': [], 'diffusion_time': [],
        'preproc_time': [], 'train_times': [],
        'inf_times': [],
    }

    seeds = range(args.N_RUNS)

    for i, seed in enumerate(seeds):
        try:
            model.reset_parameters()
        except:
            pass

        print(f'RUN #{i}: seed={seed}')
        set_seeds(seed)

        data = prep_data(osp.join(os.getcwd(), 'data'),
                         args.DATASET, args.HOPS)
        data, filter_time, diffusion_time, transform_time = transform_data(
            data, args)

        if args.VERBOSE == 1:
            print('Filter Time: {:0.4f}'.format(filter_time))
            print('Diffusion Time: {:0.4f}'.format(diffusion_time))
            print('Total Transformation Time: {:0.4f}'.format(transform_time))

        train_loader = create_loader(data, 'train', args.BATCH_SIZE)
        eval_loader = create_loader(data, 'all', args.BATCH_SIZE)

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

                if args.VERBOSE == 1:
                    print('Epoch {}:, Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(
                        epoch, train_f1, val_f1, test_f1))

        # print store best values of run
        run_results = {
            'filter_time': [filter_time],
            'diffusion_time': [diffusion_time],
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

        del data, model, train_loader, eval_loader, optimizer, evaluator, epoch_train_times, epoch_inf_times

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # print final numbers reported in thesis
    print('\n\n')
    print('==================================================')
    print('Model Parameters: {}'.format(model_params))
    print()

    print('Avg. Filter Time (s): {:.4f} +/- {:.4f}'.format(
        np.mean(results['filter_time']), np.std(results['filter_time'])))
    print('Avg. Diffusion Time (s): {:.4f} +/- {:.4f}'.format(
        np.mean(results['diffusion_time']), np.std(results['diffusion_time'])))
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

    # inputs
    parser = argparse.ArgumentParser(description='inputs')
    parser.add_argument('--VERBOSE', type=strtobool, default=False,
                        help='print epoch info if True')
    parser.add_argument('--DATASET', type=str, default='cora',
                        help='name of dataset')
    parser.add_argument('--ATTN_FILTER', type=str, default=None,
                        help='None, gat, cosine, dot_product')
    parser.add_argument('--EPOCHS', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--EVAL_EVERY', type=int, default=5,
                        help='evaluate model every _ epochs')
    parser.add_argument('--N_RUNS', type=int, default=5,
                        help='number of runs')
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
    parser.add_argument('--FILTER_BATCH_SIZE', type=int, default=100000,
                        help='for batch processing attention weights')
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

    runs(args)


# python main.py --VERBOSE 0 --DATASET 'cora' --ATTN_FILTER 'cosine' --EPOCHS 4 --EVAL_EVERY 2 --N_RUNS 1 --HOPS 3 --BATCH_SIZE 512 --LEARNING_RATE 1e-3 --WEIGHT_DECAY 1e-7 --INCEPTION_LAYERS 3 --INCEPTION_UNITS 256 --CLASSIFICATION_LAYERS 2 --CLASSIFICATION_UNITS 512 --FEATURE_DROPOUT 0.3 --NODE_DROPOUT 0.2 --BATCH_NORMALIZATION 1 --ATTN_NORMALIZATION 1
