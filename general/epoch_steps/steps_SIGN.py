import torch
import torch.nn.functional as F

from general.utils import resources  # wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@resources
def training_step(model, data, optimizer, loader):
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
        delta_mem:      from wrapper
    """
    model.train()

    total_examples = total_loss = total_correct = 0

    for idx in loader:

        # organize data
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # forward pass
        out = model(xs)
        loss = F.nll_loss(out, y)

        batch_size = idx.numel()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_correct += sum(out.argmax(dim=-1) == y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        'loss': total_loss/total_examples,
        'f1': total_correct/total_examples,
    }


@resources
@torch.no_grad()
def testing_step(model, data, loader):
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
        delta_mem:      from wrapper
    """
    model.eval()

    total_examples = total_loss = total_correct = 0

    for idx in loader:

        # organize data
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # forward pass
        out = model(xs)
        loss = F.nll_loss(out, y)

        batch_size = idx.numel()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_correct += sum(out.argmax(dim=-1) == y)

    return {
        'loss': total_loss/total_examples,
        'f1': total_correct/total_examples,
    }
