import torch
import torch.nn.functional as F

from general.utils import resources  # wrapper

### Always perform on CPU ###


@resources
def training_step(model, data, optimizer):
    """ Perform forward and backward pass
    Args:
        model:      trained GAT model
        data:       data object
        loss_fn:    loss function
        optimizer:  optimizer object

    Returns:
        train_loss:     loss @ epoch
        train_f1:       f1 @ epoch
        delta_time:     from wrapper
        delta_mem:      from wrapper
    """
    model.train()
    mask = data.train_mask

    # forward pass
    logits = model(
        data.x,
        data.edge_index
    )

    # store metrics
    mask_logits = logits[mask]
    mask_y = data.y[mask]
    mask_labels = mask_logits.argmax(dim=-1)

    loss = F.nll_loss(mask_logits, mask_y)
    f1 = sum(mask_labels == mask_y)/mask.sum()

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss,
        'f1': f1
    }


@resources
@torch.no_grad()
def testing_step(model, data, mask, logits=None):
    """ Document validation/testing loss and accuracy
    Args:
        model:      trained GAT model
        data:       data object
        mask:       val or test sets
        logits:     only predict once, use logits if exist 

    Returns:
        loss:       loss @ epoch
        f1:         f1 @ epoch
        logits:      
    """
    model.eval()

    # only predict if not done so
    if isinstance(logits, type(None)):
        logits = model(data.x, data.edge_index)

    # store metrics
    mask_logits = logits[mask]
    mask_y = data.y[mask]
    mask_labels = mask_logits.argmax(dim=-1)

    loss = F.nll_loss(mask_logits, mask_y)
    f1 = sum(mask_labels == mask_y)/mask.sum()

    outputs = {
        'loss': float(loss),
        'f1': float(f1)
    }

    return outputs, logits
