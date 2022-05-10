import wandb
import torch

from general.models.SIGN import net as SIGN
from general.utils import set_seeds, build_DataLoader, build_optimizer, build_scheduler
from general.epoch_steps.steps_SIGN import training_step, testing_step


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

    # data
    path = f'data/{config.dataset}_sign_k{config.K}.pth'
    data = torch.load(path)
    train_dl, val_dl, test_dl = build_DataLoader(data, config.batch_size)

    # model
    model = SIGN(
        data.x.shape[1],       # in_channel
        len(data.y.unique()),  # out_channel
        config.hidden_channel,
        config.dropout,
        config.K,
        config.batch_norm).to(device)

    # log number of trainable parameters
    wandb.log({
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    })

    optimizer = build_optimizer(
        model,
        config.optimizer_type,
        config.optimizer_lr,
        config.optimizer_decay
    )

    scheduler = build_scheduler(optimizer)

    # train & evaluate
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    for epoch in range(config.epochs):
        train_output, train_time, train_mem = training_step(model, data, optimizer, train_dl)
        val_output, val_time, val_mem = testing_step(model, data, val_dl)
        test_output, test_time, test_mem = testing_step(model, data, test_dl)

        scheduler.step(val_output['loss'])

        log_dict = {'epoch': epoch}
        log_dict.update({'epoch-train_'+k: v for k, v in train_output.items()})
        log_dict.update({'epoch-val_'+k: v for k, v in val_output.items()})
        log_dict.update({'epoch-test_'+k: v for k, v in test_output.items()})
        
        log_dict.update({
            'epoch-train_time': train_time,
            'epoch-val_time': val_time,
            'epoch-test_time': test_time,
            'epoch-train_mem': train_mem,
            'epoch-val_mem': val_mem,
            'epoch-test_mem': test_mem,
        })

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
