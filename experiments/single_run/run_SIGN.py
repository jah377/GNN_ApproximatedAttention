import torch
import argparse
from torch.utils.data import Dataset

# parser.add_arg doesn't play nice with bool input
from distutils.util import strtobool

from general.models.SIGN import net as SIGN
from general.utils import set_seeds, build_DataLoader, build_optimizer, build_scheduler
from general.epoch_steps.steps_SIGN import training_step, testing_step


# product: https://arxiv.org/pdf/2004.11198v2.pdf
parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='products')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--optimizer_type', type=str, default='Adam')
parser.add_argument('--optimizer_lr', type=float, default=0.0001)
parser.add_argument('--optimizer_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--hidden_channel', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--batch_norm', type=strtobool, default=True)
parser.add_argument('--batch_size', type=int, default=4096)
# only for SIGN+DotProductAttention
parser.add_argument('--attn_heads', type=int, default=None)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    set_seeds(args.seed)

    # data
    path = f'data/{args.dataset}_sign_k{args.K}.pth'
    dataset = torch.load(path)
    train_dl, val_dl, test_dl = build_DataLoader(
        dataset, args.batch_size, dataset_name=args.dataset)

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    data = dataset[0]
    del dataset

    # model
    model = SIGN(
        num_features,       # in_channel
        num_classes,        # out_channel
        args.hidden_channel,
        args.dropout,
        args.K,
        args.batch_norm).to(device)

    optimizer = build_optimizer(
        model,
        args.optimizer_type,
        args.optimizer_lr,
        args.optimizer_decay
    )

    scheduler = build_scheduler(optimizer)

    # train & evaluate
    previous_loss = 1e10
    patience = 5
    trigger_times = 0

    for epoch in range(args.epochs):
        train_output, train_resources = training_step(
            model,
            data,
            optimizer,
            train_dl
        )

        val_output, val_resources = testing_step(model, data, val_dl)
        test_output, test_resources = testing_step(model, data, test_dl)

        scheduler.step(val_output['loss'])

        # print results
        if (epoch == 0) or (epoch % 5 == 0):
            train_acc = train_output['f1']
            val_acc = val_output['f1']
            test_acc = test_output['f1']

            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

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


if __name__ == '__main__':
    main(args)
