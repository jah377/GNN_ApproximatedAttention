import torch
import torch.nn.functional as F

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
    """
    model.train()

    total_nodes = total_loss = 0

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

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(total_loss/total_nodes)


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

    train_f1 = (logits[data.train_mask].argmax(dim=-1) ==
                data.y[data.train_mask]).numpy().mean()
    val_f1 = (logits[data.val_mask].argmax(dim=-1) ==
              data.y[data.val_mask]).numpy().mean()
    test_f1 = (logits[data.test_mask].argmax(dim=-1) ==
               data.y[data.test_mask]).numpy().mean()
    val_loss = float(F.nll_loss(
        logits[data.val_mask].cpu(), data.y[data.val_mask].cpu()))

    return train_f1, val_f1, test_f1, val_loss, inf_time
