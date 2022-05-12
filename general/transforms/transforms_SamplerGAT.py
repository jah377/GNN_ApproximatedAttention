import wandb
import copy

import torch
from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborLoader

from general.models.SamplerGAT import net as GAT
from general.utils import resources  # wrapper
from general.utils import build_optimizer, build_scheduler
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


def extract_attention(data, GATtransform_params):
    """ calculate dotproduct attention
    Args:
        x:                      feature embeddings [n nodes x emb]
        edge_index:             connections
        GATtransform_params:     number of attention heads

    Returns:
        SparseTensor containing attention weights
    """
    # create loaders
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[-1]*GATtransform_params['nlayers'],
        shuffle=True,
        batch_size=GATtransform_params['batch_size'],
    )

    subgraph_loader = NeighborLoader(
        copy.copy(data),
        input_nodes=None,
        num_neighbors=[-1],
        shuffle=False,
        batch_size=GATtransform_params['batch_size'],
    )

    # no need to maintain features during evaluation
    del subgraph_loader.data.x, subgraph_loader.data.y

    # add global node index information
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

    # model
    model = GAT(
        data.x.shape[1],       # in_channel
        len(data.y.unique()),  # out_channel
        GATtransform_params['hidden_channel'],
        GATtransform_params['dropout'],
        GATtransform_params['nlayers'],
        GATtransform_params['heads_in'],
        GATtransform_params['heads_out'],
    ).to(device)

    # log number of trainable parameters
    log_dict = {
        'precomp-trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    optimizer = build_optimizer(
        model,
        GATtransform_params['optimizer_type'],
        GATtransform_params['optimizer_lr'],
        GATtransform_params['optimizer_decay']
    )

    scheduler = build_scheduler(optimizer)

    # train & evaluate
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    for epoch in range(GATtransform_params['epochs']):

        train_output, train_resources = training_step(
            model,
            optimizer,
            train_loader
        )

        val_output, logits, val_resources = testing_step(
            model,
            data,
            subgraph_loader,
            data.val_mask,
            logits=None,  # must model.inference
        )

        test_output, _, test_resources = testing_step(
            model,
            data,
            subgraph_loader,
            data.test_mask,
            logits=logits  # use prev. pred
        )

        scheduler.step(val_output['loss'])

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

        # early stopping
        current_loss = val_output['loss']
        if current_loss > previous_loss:
            trigger_times += 1
            if trigger_times >= patience:
                break
        else:
            trigger_times = 0
        previous_loss = current_loss

    # get attention SparseTensor
    model.eval()
    attn_sparse = model.extract_features(data.x, subgraph_loader)

    return attn_sparse
