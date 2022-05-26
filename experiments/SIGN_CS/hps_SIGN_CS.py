import wandb
import os.path as osp

import torch
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_SIGN import SIGN
from steps_SIGN import train_epoch, test_epoch
from transform_CosineSimilarity import CosineAttention
from general.utils import set_seeds, standardize_data, create_loader

hyperparameter_defaults = dict(
    dataset='cora',
    seed=42,
    optimizer_type='Adam',
    optimizer_lr=1e-3,
    optimizer_decay=1e-3,
    epochs=5,
    hidden_channel=256,
    dropout=0.6,
    K=1,
    batch_norm=1,
    batch_size=256,
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config):
    set_seeds(config.seed)

    # IMPORT & STANDARDIZE DATA
    path = f'data/{config.dataset}_sign_k{config.K}.pth'
    transform_path = f'data/{config.dataset}_sign_cs_transformed.pth'

    if not osp.isfile(transform_path):
        data = standardize_data(torch.load(path), config.dataset)
        data, transform_time = CosineAttention(
            data,
            config.K,
            config.cs_batch_size,
        )

        wandb.log({'precomp-transform_time': transform_time})

        torch.save(data, transform_path)
        print('\n~~~ TRANSFORM PERFORMED ~~~\n')
        print(data)
    else:
        data = torch.load(transform_path)   # already standardized
        assert hasattr(data, 'edge_index')  # must be torch data object

    # BUILD DATALOADER
    train_loader = create_loader(data, 'train', batch_size=config.batch_size)
    val_loader = create_loader(data, 'val', batch_size=config.batch_size)
    test_loader = create_loader(data, 'test', batch_size=config.batch_size)

    # BUILD MODEL
    model = SIGN(
        data.num_features,  # in_channel
        data.num_classes,   # out_channel
        config.hidden_channel,
        config.dropout,
        config.K,
        config.batch_norm).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({'trainable_params': n_params})  # log number of params

    # BUILD OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_decay
    )

    # BUILD SCHEDULER (modulates learning rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',     # nll_loss expected to decrease over epochs
        factor=0.1,     # lr reduction factor
        patience=5,     # reduce lr after _ epochs of no improvement
        min_lr=1e-6,    # min learning rate
        verbose=False,  # do not monitor lr updates
    )

    # EVALUATOR
    if data.dataset_name in ['products', 'arxiv']:
        evaluator = Evaluator(name=f'ogbn-{data.dataset_name}')
    else:
        evaluator = None

    # RUN THROUGH EPOCHS
    # params for early termination
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    for epoch in range(config.epochs):

        training_out, training_time = train_epoch(
            model, data, optimizer, train_loader)

        train_out = test_epoch(model, data, train_loader, evaluator)
        val_out = test_epoch(model, data, val_loader, evaluator)
        test_out = test_epoch(model, data, test_loader, evaluator)

        scheduler.step(val_out['loss'])  # modulate learning rate

        # log results
        log_dict = {
            'epoch': epoch,
            'epoch-training-train_time': training_time
        }

        log_dict.update(
            {'epoch-train-train_'+k: v for k, v in training_out.items()}
        )

        log_dict.update(
            {'epoch-eval-train_'+k: v for k, v in train_out.items()}
        )

        log_dict.update(
            {'epoch-eval-val_'+k: v for k, v in val_out.items()}
        )

        log_dict.update(
            {'epoch-eval-test_'+k: v for k, v in test_out.items()}
        )
        wandb.log(log_dict)

        # early stopping
        current_loss = val_out['loss']
        if current_loss > previous_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('$$$ EARLY STOPPING TRIGGERED $$$')
                break
        else:
            trigger_times = 0
        previous_loss = current_loss


if __name__ == "__main__":
    main(config)
