import wandb

import torch
from torch_sparse import SparseTensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from general.models.FullBatchGAT import net as GAT
from general.utils import resources  # wrapper
from general.epoch_steps.steps_FullBatchGAT import training_step, testing_step


### Always perform on CPU ###


@resources
def transform_wAttention(data, dataset: str, K: int, GATtransform_params):
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
    adj_t = extract_attention(data, GATtransform_params.get(dataset))
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # =========== not part of T.SIGN(K) ===========

    assert data.x is not None
    xs = [data.x]

    for i in range(1, K + 1):

        xs += [adj_t @ xs[-1]]
        data[f'x{i}'] = xs[-1]

    assert hasattr(data, f'x{K}')
    return data


def extract_attention(data, GATdict):
    """ calculate dotproduct attention
    Args:
        x:                      feature embeddings [n nodes x emb]
        edge_index:             connections
        GATtransform_params:     number of attention heads

    Returns:
        SparseTensor containing attention weights
    """
    cpu = torch.device('cpu')

    # BUILD MODEL
    model = GAT(
        data.num_features,  # in_channel
        data.num_classes,  # out_channel
        GATdict['hidden_channel'],
        GATdict['dropout'],
        GATdict['nlayers'],
        GATdict['heads_in'],
        GATdict['heads_out'],
    ).to(cpu)

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

        train_out, train_resources = training_step(model, data, optimizer)
        test_out = testing_step(model, data)

        scheduler.step(test_out['val_loss'])

        # log results
        s = 'precomp-epoch'
        log_dict = {f'{s}': epoch}
        log_dict.update({f'{s}-'+k: v for k, v in train_out.items()})
        log_dict.update({f'{s}-train-'+k: v for k,
                        v in train_resources.items()})
        log_dict.update({f'{s}-'+k: v for k, v in test_out.items()})
        wandb.log(log_dict)

        # early stopping
        current_loss = test_out['val_loss']
        if current_loss > previous_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('~~~ early stop triggered ~~~')
                break
        else:
            trigger_times = 0
        previous_loss = current_loss

    # extract attention
    model.eval()
    _, (attn_i, attn_w) = model.extract_features(
        data.x,
        data.edge_index,
        return_attention_weights=True)

    dim = data.num_nodes  # number of nodes

    return SparseTensor(
        row=attn_i[0],  # edge_indices (row)
        col=attn_i[1],  # edge indices (col0)
        value=attn_w.mean(axis=1).detach(),
        sparse_sizes=(dim, dim)
    )  # to replace adj_t
