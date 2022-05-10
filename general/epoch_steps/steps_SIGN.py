import torch
import torch.nn.functional as F

from general.utils import resources  # wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



@resources
def training_step(model, data, optimizer, loader):
    """ Perform forward and backward pass on SIGN
    https://torchmetrics.readthedocs.io/en/latest/pages/overview.html?highlight=collection#metriccollection

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

    cum_n = cum_loss = cum_correct = 0

    for idx in loader:

        # organize data
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # forward pass
        batch_logits = model(xs)
        batch_labels = batch_logits.argmax(dim=-1)
        batch_loss = F.nll_loss(batch_logits, y)

        # store
        cum_n += idx.numel()
        cum_loss += float(batch_loss) * cum_n
        cum_correct += sum(batch_labels == y)

        # backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return {
        'loss': cum_loss/cum_n,
        'f1': cum_correct/cum_n,
    }

@torch.no_grad()
@resources
def testing_step(model, data, loader):
    """ Document validation or test loss and accuracy
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

    cum_n = cum_loss = cum_correct = 0

    for idx in loader:

        # organize
        xs = [data.x[idx].to(device)]           # add x[idx] to device
        xs += [data[f'x{i}'][idx].to(device)
               for i in range(1, model.K + 1)]  # add each A^K*X[idx] to xs
        y = data.y[idx].to(device)              # move target to device

        # predict
        batch_logits = model(xs)
        batch_labels = batch_logits.argmax(dim=-1)
        batch_loss = F.nll_loss(batch_logits, y)

        # store
        cum_n += idx.numel()
        cum_loss += float(batch_loss) * cum_n
        cum_correct += sum(batch_labels == y)

    return {
        'loss': cum_loss/cum_n,
        'f1': cum_correct/cum_n,
    }
