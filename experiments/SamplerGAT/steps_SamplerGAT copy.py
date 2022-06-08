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

        # f1 score
        if evaluator:
            f1_score = evaluator.eval({
                "y_true": torch.cat(logits[mask].argmax(dim=-1), dim=0).unsqueeze(dim=1),
                "y_pred": torch.cat(data.y[mask], dim=0).unsqueeze(dim=1),
            })['acc']
        else:
            f1_score = (logits[mask].argmax(dim=-1) ==
                        data.y[mask]).numpy().mean()

        output.update({
            f'{split}_loss': float(F.nll_loss(logits[mask].cpu(), data.y[mask].cpu())),
            f'{split}_f1': f1_score
        })

    return output, inf_time
