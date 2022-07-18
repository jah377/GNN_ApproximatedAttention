import glob
import wandb
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from SIGNff_utils import set_seeds, time_wrapper, create_loader, create_evaluator_fn
from transformation import transform_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparameter_defaults = dict(
    DATASET='cora',
    SEED=42,
    EPOCHS=300,
    HOPS=3,
    BATCH_SIZE=4028,
    LEARNING_RATE=1e-3,
    WEIGHT_DECAY=1e-3,
    INCEPTION_LAYERS=2,
    INCEPTION_UNITS=512,
    CLASSIFICATION_LAYERS=2,
    CLASSIFICATION_UNITS=512,
    FEATURE_DROPOUT=0.1,
    NODE_DROPOUT=0.5,
    BATCH_NORMALIZATION=1,
    # ATTN_HEADS=4,
    # DPA_NORMALIZATION=1,
    # CS_BATCH_SIZE=10000,
    LR_PATIENCE=5,
    TERMINATION_PATIENCE=20
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeedForwardNet(nn.Module):
    """
    https://github.com/THUDM/CRGNN/blob/main/layer.py
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/sign/sign.py
    """

    def __init__(
        self,
        in_units: int,
        out_units: int,
        hidden_channels: int,
        node_dropout: float,
        n_layers: int,
        batch_norm: bool = True
    ):
        super(FeedForwardNet, self).__init__()
        self.n_layers = max(1, n_layers)
        self.batch_norm = batch_norm
        self.prelu = nn.PReLU()
        self.node_dropout = nn.Dropout(node_dropout)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if self.n_layers == 1:
            self.lins.append(nn.Linear(in_units, out_units))
        else:
            self.lins.append(nn.Linear(in_units, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            for _ in range(self.n_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            self.lins.append(nn.Linear(hidden_channels, out_units))

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight, gain=gain)
            nn.init.zeros_(lin.bias)
        for bns in self.bns:
            bns.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.lins):
            x = layer(x)
            if i < self.n_layers-1:
                if self.batch_norm == True:
                    x = self.node_dropout(self.prelu(self.bns[i](x)))
                else:
                    x = self.node_dropout(self.prelu(x))
        return x


class SIGN(torch.nn.Module):
    def __init__(
        self,
        in_units: int,                  # feature length
        out_units: int,                 # num classes
        inception_units: int,           # hidden units
        inception_layers: int,          # n layers
        classification_units: int,      # hidden units
        classification_layers: int,     # n layers
        feature_dropout: float,         # input dropout
        node_dropout: float,            # nn regularization
        hops: int,                      # K-hop aggregation
        batch_norm: bool = True         # include batch normalization
    ):
        super(SIGN, self).__init__()
        self.hops = hops
        self.inception_layers = inception_layers
        self.classification_layers = classification_layers
        self.batch_norm = batch_norm
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.node_dropout = nn.Dropout(node_dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()

        # inception feedforward layers
        for _ in range(self.hops + 1):
            self.inception_ffs.append(
                FeedForwardNet(
                    in_units,
                    inception_units,
                    inception_units,
                    node_dropout,
                    inception_layers,
                    batch_norm
                )
            )

        # feedforward layer for concatenated outputs
        self.concat_ff = FeedForwardNet(
            (self.hops+1)*inception_units,
            out_units,
            classification_units,
            node_dropout,
            classification_layers,
            batch_norm
        )

    def reset_parameters(self):
        for layer in self.inception_ffs:
            layer.reset_parameters()
        self.concat_ff.reset_parameters()

    def forward(self, xs):
        """ xs = [AX, A(AX), ..., AX^K] """

        xs = [self.feature_dropout(x) for x in xs]
        hs = []

        for hop, layer in enumerate(self.inception_ffs):
            hs.append(layer(xs[hop]))

        return self.concat_ff(
            self.node_dropout(
                self.prelu(
                    torch.cat(hs, dim=-1)
                ))).log_softmax(dim=-1)


@time_wrapper
def train(data, model, optimizer, train_loader):
    """ return avg. loss and training time """
    model.train()

    total_examples = total_loss = 0
    for batch in train_loader:
        xs = [data.x[batch].to(device)]
        xs += [data[f'x{k}'][batch].to(device) for k in range(1, model.hops+1)]
        loss = F.nll_loss(model(xs), data.y[batch].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = int(batch.numel())
        total_examples += int(batch_size)
        total_loss += float(loss) * batch_size

    return total_loss/total_examples


@time_wrapper
def inference(model, xs):
    """ return logits and inference time """
    return model(xs)


@torch.no_grad()
def eval(data, model, loader, evaluator):
    model.eval()

    inf_time = 0
    total_examples = total_loss = 0
    preds, labels = [], []
    for batch in loader:
        xs = [data.x[batch].to(device)]
        xs += [data[f'x{hop}'][batch].to(device)
               for hop in range(1, model.hops + 1)]
        out, batch_time = inference(model, xs)
        loss = F.nll_loss(out, data.y[batch].to(out.device))

        batch_size = int(batch.numel())
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        inf_time += batch_time

        labels.append(data.y[batch].cpu())
        preds.append(out.argmax(dim=-1).cpu())

    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    loss = total_loss/total_examples
    f1 = evaluator(preds, labels)

    return loss, f1, inf_time


def main(config):
    """ perform sweep """
    assert config.TERMINATION_PATIENCE > config.LR_PATIENCE, 'Termination patience cannot be less than learning rate patience'

    set_seeds(config.SEED)

    # import data
    file_name = f'{config.DATASET}_sign_k0.pth'
    file_path = glob.glob(f'./**/{file_name}', recursive=True)[0][2:]
    folder_path = osp.dirname(file_path)
    transform_path = osp.join(
        folder_path, f'{config.DATASET}_k{config.HOPS}_{config.TRANSFORMATION}.pth')

    if not osp.isfile(transform_path):
        data = torch.load(file_path)
        data, _ = transform_data(data, config)
        torch.save(data, transform_path)
        print()
        print(f'TRANSFORMED FILE: {transform_path}')
        print()
    else:
        data = torch.load(transform_path)
        assert hasattr(data, 'edge_index')  # sanity check

    # build dataloader
    train_loader = create_loader(data, 'train', batch_size=config.BATCH_SIZE)
    val_loader = create_loader(data, 'val', batch_size=config.BATCH_SIZE)

    # build model
    model = SIGN(
        data.num_features,
        data.num_classes,
        config.INCEPTION_UNITS,
        config.INCEPTION_LAYERS,
        config.CLASSIFICATION_UNITS,
        config.CLASSIFICATION_LAYERS,
        config.FEATURE_DROPOUT,
        config.NODE_DROPOUT,
        config.HOPS,
        config.BATCH_NORMALIZATION,
    ).to(device)

    # build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # build scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',     # nll_loss expected to decrease over epochs
        factor=0.1,     # lr reduction factor
        patience=config.LR_PATIENCE,     # reduce lr after _ epochs of no improvement
        min_lr=1e-6,    # min learning rate
        verbose=False,  # do not monitor lr updates
    )

    # build evaluator
    evaluator = create_evaluator_fn(config.DATASET)

    # train and evaluation
    previous_loss, trigger_times = 1e10, 0
    for epoch in range(1, config.EPOCHS+1):
        _, training_time = train(data, model, optimizer, train_loader)

        train_loss, train_f1, _ = eval(
            data, model, train_loader, evaluator)
        val_loss, val_f1, _ = eval(
            data, model, val_loader, evaluator)

        scheduler.step(val_loss)

        wandb.log({
            'epoch': epoch,
            'training_time': training_time,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_f1': val_f1,
        })

        # early stopping
        current_loss = val_loss
        if current_loss > previous_loss:
            trigger_times += 1
            if trigger_times >= config.TERMINATION_PATIENCE:
                print('$$$ EARLY STOPPING TRIGGERED $$$')
                break
        else:
            trigger_times = 0
        previous_loss = current_loss


if __name__ == '__main__':
    print(config)
    main(config)
