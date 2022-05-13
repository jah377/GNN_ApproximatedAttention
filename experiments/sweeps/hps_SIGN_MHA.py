import wandb
import os.path as osp

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from general.models.SIGN import net as SIGN
from general.utils import set_seeds, standardize_dataset, build_DataLoader
from general.epoch_steps.steps_SIGN import training_step, testing_step

from general.transforms.transforms_DotProduct import transform_wAttention


#################################################################
########## THIS SHOULD BE IDENTICAL TO HPS_SIGN_SHA.PY ##########
#################################################################

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
    attn_heads=1,
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config):
    set_seeds(config.seed)

    # IMPORT & STANDARDIZE DATA
    path = f'data/{config.dataset}_sign_k0.pth'
    transform_path = f'data/{config.dataset}_sign_k0_heads{config.attn_heads}_transformed.pth'

    if not osp.isfile(transform_path):
        data = standardize_dataset(torch.load(path), config.dataset)
        data, trans_resources = transform_wAttention(
            data,
            config.K,
            config.attn_heads
        )

        wandb.log({'precomp-transform_'+k: v for k,
                  v in trans_resources.items()})

        torch.save(data, transform_path)
        print('\n~~~ TRANSFORM PERFORMED ~~~\n')
    else:
        data = torch.load(transform_path)  # already standardized

    # BUILD DATALOADER
    train_dl, val_dl, test_dl = build_DataLoader(
        data,
        config.batch_size,
        dataset_name=config.dataset
    )

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
        config.optimizer_lr,
        config.optimizer_decay
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

    # RUN THROUGH EPOCHS
    # params for early termination
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    for epoch in range(config.epochs):

        train_output, train_resources = training_step(
            model, data, optimizer, train_dl)
        val_output, val_resources = testing_step(model, data, val_dl)
        test_output, test_resources = testing_step(model, data, test_dl)

        scheduler.step(val_output['loss'])  # modulate learning rate

        # log results
        log_dict = {'epoch': epoch}
        log_dict.update({'epoch-train_'+k: v for k, v in train_output.items()})
        log_dict.update({'epoch-val_'+k: v for k, v in val_output.items()})
        log_dict.update({'epoch-test_'+k: v for k, v in test_output.items()})
        log_dict.update({'epoch-train_'+k: v for k,
                        v in train_resources.items()})
        log_dict.update({'epoch-val_'+k: v for k, v in val_resources.items()})
        log_dict.update({'epoch-test_'+k: v for k,
                        v in test_resources.items()})
        wandb.log(log_dict)

        # early stopping
        current_loss = val_output['loss']
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
