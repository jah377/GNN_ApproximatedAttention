import torch
import torch.nn.functional as F

from general.utils import time_wrapper  # wrapper

from ogb.nodeproppred import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@time_wrapper
def train_epoch(model, data, optimizer, loader):
    """ Perform forward and backward pass on SIGN
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sign.py
    Args:
        model:      SIGN model
        data:       data object
        loader:     DataLoader of train/val/test set
        optimizer:  optimizer object

    Returns:
        train_loss:     loss @ epoch
        train_f1:       f1 @ epoch
        delta_time:     from wrapper
    """
    model.train()

    total_examples = total_correct = total_loss = 0
    for idx in loader:

        # organize data
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # forward pass
        out = model(xs)
        loss = F.nll_loss(out, y)

        batch_size = int(idx.numel())
        total_examples += int(batch_size)
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        'f1': total_correct/total_examples,
        'loss': total_loss/total_examples,
    }


@torch.no_grad()
def test_epoch(model, data, loader):
    """ Document validation or test loss and accuracy
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sign.py

    Args:
        model:      trained GAT model
        data:       data object
        loader:     train, val, or test DataLoader

    Returns:
        loss:       loss @ epoch
        f1:         f1 @ epoch
        delta_time:     from wrapper
    """
    model.eval()

    if data.dataset_name in ['products', 'arxiv']:
        evaluator = Evaluator(name=f'ogbn-{data.dataset_name}')

    @time_wrapper
    def predict(model, xs):
        return model(xs)

    total_time = total_examples = total_loss = 0
    y_pred, y_true = [], []

    for idx in loader:

        # organize data
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # forward pass
        out, out_time = predict(model, xs)
        loss = F.nll_loss(out, y)

        y_pred.append(out.argmax(dim=1).flatten().cpu())
        y_true.append(y.cpu())

        # store
        batch_size = int(idx.numel())
        total_time += out_time
        total_examples += batch_size
        total_loss += float(loss) * batch_size

        # f1 score
        if data.dataset_name in ['products', 'arxiv']:
            f1_score = evaluator.eval({
                "y_true": torch.cat(y_true, dim=0),
                "y_pred": torch.cat(y_pred, dim=0),
            })['acc']
        else:
            f1_score = (y_pred == y_true).mean()

    return {
        'f1': f1_score,
        'loss': total_loss/total_examples,
        'time': total_time,  # total inference time
    }