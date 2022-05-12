import argparse
import torch

from general.models.FullBatchGAT import net as GAT
from general.utils import build_optimizer, build_scheduler
from general.epoch_steps.steps_FullBatchGAT import training_step, testing_step
from general.best_configs.FullBatchGAT_configs import params_dict as GATtransform_params

parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--dataset', type=str, default='pubmed')
args = parser.parse_args()


def main(args):

    # import or transform data
    path = f'data/{args.dataset}_sign_k0.pth'
    data = torch.load(path)

    # model
    model = GAT(
        data.x.shape[1],       # in_channel
        len(data.y.unique()),  # out_channel
        GATtransform_params['hidden_channel'],
        GATtransform_params['dropout'],
        GATtransform_params['nlayers'],
        GATtransform_params['heads_in'],
        GATtransform_params['heads_out'],
    )

    optimizer = build_optimizer(
        model,
        GATtransform_params['optimizer_type'],
        GATtransform_params['optimizer_lr'],
        GATtransform_params['optimizer_decay']
    )

    scheduler = build_scheduler(optimizer)

    # train & evaluate
    for epoch in range(GATtransform_params['epochs']):

        # perform training and testing step
        train_output, train_resources = training_step(
            model,
            data,
            optimizer
        )

        val_output, logits, val_resources = testing_step(
            model,
            data,
            data.val_mask,
            logits=None,
        )

        test_output, _, test_resources = testing_step(
            model,
            data,
            data.test_mask,
            logits=logits,
        )

        scheduler.step(val_output['loss'])  # dynamic learning rate

        # print results
        if (epoch == 0) or (epoch % 5 == 0):
            train_acc = train_output['f1']
            val_acc = val_output['f1']
            test_acc = test_output['f1']

            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')


if __name__ == '__main__':
    main(args)
