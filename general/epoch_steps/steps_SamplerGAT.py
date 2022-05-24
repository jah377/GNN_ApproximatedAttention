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
        'train_loss': float(total_loss/total_nodes),
        'train_f1': float(total_correct/total_nodes),
    }

    return outputs


@torch.no_grad()
def test_epoch(model, data, subgraph_loader):
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

    logits, inf_resources = model.inference(data.x, subgraph_loader)
    output = {f'inf_{k}': v for k, v in inf_resources.items()}

    for split in ['train', 'val']:

        mask = eval(f'data.{split}_mask')
        mask_logits = logits[mask]
        mask_yhat = mask_logits.argmax(dim=-1)
        mask_y = data.y[mask].to(mask_logits.device)

        output.update({
            f'{split}_loss': F.nll_loss(mask_logits, mask_y).item(),
            f'{split}_f1': (sum(mask_yhat == mask_y)/len(mask)).item()
        })

    return output
