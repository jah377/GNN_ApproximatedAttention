import copy

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ogb.nodeproppred import Evaluator

from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborLoader

from model_SamplerGAT import GAT
from general.utils import time_wrapper  # wrapper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@time_wrapper
def train_epoch(model, optimizer, train_loader):
    """ Perform forward and backward pass
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py

    Args:
        model:          GAT_loader model
        optimizer:      optimizer object
        train_loader:   contains the data

    Returns:
        train_loss:     loss @ epoch
        delta_time:     from wrapper
        delta_mem:      from wrapper
    """
    model.train()

    total_nodes = total_correct = total_loss = 0

    for batch in train_loader:
        batch_size = batch.batch_size

        # forward pass
        logits = model(
            batch.x.to(device),
            batch.edge_index.to(device)
        )[:batch_size]

        y = batch.y[:batch_size].to(logits.device)
        loss = F.nll_loss(logits, y)

        # store metrics
        total_nodes += batch_size
        total_loss += float(loss) * batch_size
        total_correct += int(sum(logits.argmax(dim=-1) == y))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    outputs = {
        'loss': float(total_loss/total_nodes),
        'f1': float(total_correct/total_nodes),
    }

    return outputs


@torch.no_grad()
def test_epoch(model, data, subgraph_loader, evaluator=None):
    """ Perform forward and backward pass
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py

    Args:
        model:              GAT_loader model
        data:               data
        subgraph_loader:    contain batch indices

    Returns:
        output:
            .inf_time
            .inf_mem
            .train_loss
            .val_loss
            .train_f1
            .val_f1
    """
    model.eval()

    logits, inf_time = model.inference(data.x, subgraph_loader)

    output = {}

    for split in ['train', 'val', 'test']:

        mask = data[f'{split}_mask']

        # # f1 score
        # if evaluator:
        #     f1_score = evaluator.eval({
        #         "y_true": torch.cat(logits[mask].argmax(dim=-1).cpu(), dim=0),
        #         "y_pred": torch.cat(data.y[mask].cpu(), dim=0),
        #     })['acc']
        # else:
        #     f1_score = (torch.cat(logits[mask].argmax(dim=-1).cpu(), dim=0) ==
        #                 torch.cat(data.y[mask].cpu(), dim=0)).numpy().mean()

        output.update({
            f'{split}_loss': F.nll_loss(logits[mask].cpu(), data.y[mask].cpu()).item(),
            # f'{split}_f1': f1_score
        })

    return output, inf_time


@time_wrapper
def GATAttention(data, K: int, GATdict):
    """
    Args:
        data:           data object
        K:              number of SIGN transformations
        GATtransform_params:     dict of best hyperparameters determined from sweep

    Returns:
        data:   transformed data
        time_:  from wrapper
        mem_:   from wrapper
    """

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
    # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    # replace adj with DotProductAttention weights
    adj_t = extract_attention(data, GATdict)
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    # sanity check
    if K > 0:
        assert hasattr(data, f'x{K}')

    return data


def extract_attention(data, GATdict):
    """ train GAT model and extract attention
    Args:
        data:
        GATtransform_params:     number of attention heads

    Returns:
        SparseTensor containing attention weights
    """

    # CREATE TRAINING AND SUBGRAPH LOADERS
    # [n_neighbors] = hyperparameter
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,  # can be bool or n_id indices
        num_neighbors=[GATdict['n_neighbors']]*GATdict['nlayers'],
        shuffle=True,
        batch_size=GATdict['batch_size'],
        drop_last=True,  # remove final batch if incomplete
    )

    subgraph_loader = NeighborLoader(
        copy.copy(data),
        input_nodes=None,
        num_neighbors=[-1]*GATdict['nlayers'],  # sample all neighbors
        shuffle=False,                          # :batch_size in sequential order
        batch_size=GATdict['batch_size'],
        drop_last=False,
    )
    subgraph_loader.data.num_nodes = data.num_nodes
    del subgraph_loader.data.x, subgraph_loader.data.y  # only need indices

    # BUILD MODEL
    model = GAT(
        data.num_features,  # in_channel
        data.num_classes,  # out_channel
        GATdict['hidden_channel'],
        GATdict['dropout'],
        GATdict['nlayers'],
        GATdict['heads_in'],
        GATdict['heads_out'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # BUILD OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GATdict['optimizer_lr'],
        weight_decay=GATdict['optimizer_decay'],
    )

    # BUILD SCHEDULER (modulates learning rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',     # nll_loss expected to decrease over epochs
        factor=0.1,     # lr reduction factor
        patience=5,     # reduce lr after _ epochs of no improvement
        min_lr=1e-6,    # min learning rate
        verbose=False,  # do not monitor lr updates
    )

    # evaluator, if ogbn dataset
    if data.dataset_name in ['products', 'arxiv']:
        evaluator = Evaluator(name=f'ogbn-{data.dataset_name}')
    else:
        evaluator = None

    # RUN THROUGH EPOCHS
    for epoch in range(GATdict['epochs']):

        train_epoch(
            model,
            optimizer,
            train_loader
        )

        test_out, _ = test_epoch(model, data, subgraph_loader, evaluator)
        scheduler.step(test_out['val_loss'])

    # EXTRACT ATTENTION MATRIX
    model.eval()
    attn_sparse = model.extract_features(data.x, subgraph_loader)

    return attn_sparse
