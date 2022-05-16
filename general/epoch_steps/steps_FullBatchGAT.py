import torch
import torch.nn.functional as F

from general.utils import resources  # wrapper

### Always perform on CPU ###
cpu = torch.device('cpu')


@resources
def training_step(model, data, optimizer):
    """ Perform forward and backward pass
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py

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
    mask_yhat = mask_logits.argmax(dim=-1)
    mask_y = data.y[mask].to(mask_logits.device)
    loss = F.nll_loss(mask_logits, mask_y)

    output = {
        'train_loss': loss,
        'train_f1': sum(mask_yhat == mask_y).div(len(mask))
    }

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output


@torch.no_grad()
def testing_step(model, data):
    """ Document validation/testing loss and accuracy
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py

    Args:
        model:      trained GAT model
        data:       data object

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

    @resources
    def inference(model, data):
        return model(data.x.to(cpu), data.edge_index.to(cpu))

    logits, inf_resources = inference(model, data)
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
