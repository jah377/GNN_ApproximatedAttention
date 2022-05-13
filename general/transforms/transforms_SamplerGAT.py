import wandb
import copy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborLoader

from general.models.SamplerGAT import net as GAT
from general.utils import resources  # wrapper
from general.epoch_steps.steps_SamplerGAT import training_step, testing_step

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@resources
def transform_wAttention(data, dataset: str, K: int, GATtransform_params):
    """
    Args:
        data:           data object
        dataset:        str name of dataset
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
    adj_t = extract_attention(data, GATtransform_params.get(dataset))
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

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
        num_neighbors=[GATdict['nNeighbors']]*GATdict['nlayers'],
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
    wandb.log({'precomp-trainable_params': n_params})  # size of model

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

    # RUN THROUGH EPOCHS
    # params for early termination
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    for epoch in range(GATdict['epochs']):

        train_out, train_resources = training_step(
            model,
            optimizer,
            train_loader
        )

        test_out = testing_step(model, optimizer, data, subgraph_loader)

        val_loss = test_out['val_loss']
        scheduler.step(val_loss)

        # log results
        s = 'precomp-epoch'
        log_dict = {f'{s}': epoch}
        log_dict.update({f'{s}-'+k: v for k, v in train_out.items()})
        log_dict.update({f'{s}-train-'+k: v for k,
                        v in train_resources.items()})
        log_dict.update({f'{s}-'+k: v for k, v in test_out.items()})
        wandb.log(log_dict)

        # early stopping
        current_loss = val_loss
        if current_loss > previous_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('~~~ early stop triggered ~~~')
                break
        else:
            trigger_times = 0
        previous_loss = current_loss

    # EXTRACT ATTENTION MATRIX
    model.eval()
    attn_sparse = model.extract_features(data.x, subgraph_loader)

    return attn_sparse
