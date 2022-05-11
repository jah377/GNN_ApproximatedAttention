import wandb

import torch
from torch_sparse import SparseTensor

from general.models.FullBatchGAT import net as GAT
from general.utils import resources  # wrapper
from general.utils import build_optimizer, build_scheduler
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

    return data


def extract_attention(data, GATtransform_params):
    """ calculate dotproduct attention
    Args:
        x:                      feature embeddings [n nodes x emb]
        edge_index:             connections
        GATtransform_params:     number of attention heads

    Returns:
        SparseTensor containing attention weights
    """

    # model
    model = GAT(
        data.x.shape[1],       # in_channel
        len(data.y.unique()),  # out_channel
        GATtransform_params['hidden_channel'],
        GATtransform_params['dropout'],
        GATtransform_params['nlayers'],
        GATtransform_params['heads_in'],
        GATtransform_params['heads_out'],
    )

    # log number of trainable parameters
    wandb.log({
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    })

    optimizer = build_optimizer(
        model,
        GATtransform_params['optimizer_type'],
        GATtransform_params['optimizer_lr'],
        GATtransform_params['optimizer_decay']
    )

    scheduler = build_scheduler(optimizer)

    # train & evaluate
    for epoch in range(GATtransform_params['epochs']):

        # perform training and testing step
        train_output, train_resources = training_step(
            model,
            data,
            optimizer
        )
        val_output, logits, val_resources = testing_step(
            model,
            data,
            data.val_mask,
            logits=None,
        )
        test_output, _, test_resources = testing_step(
            model,
            data,
            data.test_mask,
            logits=logits,
        )

        scheduler.step(val_output['loss'])  # dynamic learning rate

        # log results
        s = 'precomp-epoch'
        log_dict = {f'{s}': epoch}
        log_dict.update({f'{s}-train_'+k: v for k, v in train_output.items()})
        log_dict.update({f'{s}-val_'+k: v for k, v in val_output.items()})
        log_dict.update({f'{s}-test_'+k: v for k, v in test_output.items()})

        log_dict.update({f'{s}-train_'+k: v for k,
                        v in train_resources.items()})
        log_dict.update({f'{s}-val_'+k: v for k, v in val_resources.items()})
        log_dict.update({f'{s}-test_'+k: v for k,
                        v in test_resources.items()})

        wandb.log(log_dict)

    # extract attention
    model.eval()
    _, att_weights = model.extract_features(
        data.x,
        data.edge_index,
        return_attention_weights=True)
    dim = data.x.shape[0]  # number of nodes

    return SparseTensor(
        row=att_weights[0][0],  # edge_indices (row)
        col=att_weights[0][1],  # edge indices (col0)
        value=att_weights[1].mean(axis=1).detach(),
        sparse_sizes=(dim, dim)
    )  # to replace adj_t
