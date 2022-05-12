import torch
import torch.nn.functional as F

from general.utils import resources  # wrapper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@resources
def training_step(model, optimizer, train_loader):
    """ Perform forward and backward pass
    Args:
        model:          GAT_loader model
        optimizer:      optimizer object
        train_loader:   contains the data

    Returns:
        train_loss:     loss @ epoch
        train_f1:       f1 @ epoch
        delta_time:     from wrapper
        delta_mem:      from wrapper
    """
    model.train()

    cum_n = cum_loss = cum_correct = 0

    for batch in train_loader:

        # organize data
        y = batch.y[:batch.batch_size].to(device)

        # forward pass
        logits = model(batch.x.to(device), batch.edge_index.to(device))
        batch_loss = F.nll_loss(logits[:batch.batch_size], y)

        # store metrics
        cum_n += batch.batch_size
        cum_loss += float(batch_loss) * cum_n
        cum_correct += sum(logits[:batch.batch_size].argmax(dim=-1) == y)

        # backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    outputs = {
        'loss': float(cum_loss/cum_n),
        'f1': float(cum_correct/cum_n),
    }

    return outputs


@resources
@torch.no_grad()
def testing_step(model, data, subgraph_loader, mask, logits=None):
    """ Document validation/testing loss and accuracy
    Args:
        model:              trained GAT model
        data:               data object
        subgraph_loader:    
        mask:               validation or testing mask
        logits:             only predict if not done before 

    Returns:
        loss:       loss @ epoch
        f1:         f1 @ epoch
    """
    model.eval()
    outputs = {}

    # only predict if not done so
    if isinstance(logits, type(None)):
        logits, inf_resources = model.inference(data.x, subgraph_loader)
        outputs.update({'inference_'+k: v for k, v in inf_resources.items()})

    # validation or testing metrics
    mask_logits = logits[mask].to(device)
    mask_y = data.y[mask].to(device)

    loss = F.nll_loss(mask_logits, mask_y)
    f1 = int((mask_logits.argmax(dim=-1) == mask_y).sum())

    outputs.update({
        'loss': float(loss),
        'f1': float(f1),
    })

    return outputs, logits
