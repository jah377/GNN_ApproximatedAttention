import yaml
import argparse

# parser.add_arg doesn't play nice with bool input
from distutils.util import strtobool


parser = argparse.ArgumentParser(description='inputs')

# general input arguments
parser.add_argument('--DATASET', type=str, default='cora')
parser.add_argument('--MODEL', type=str, default=None)
parser.add_argument('--METHOD', type=str, default='random')
parser.add_argument('--TRAIN_FILE', type=str, default=None)
parser.add_argument('--YAML_FILE', type=str, default=None)
parser.add_argument('--RUN_TRIAL', type=strtobool, default=True)
parser.add_argument('--N_WORKERS', type=int, default=1)
parser.add_argument('--SEED', type=int, default=None)
parser.add_argument('--OPTIMIZER_LR', type=float, default=None)
parser.add_argument('--OPTIMIZER_DECAY', type=float, default=None)
parser.add_argument('--EPOCHS', type=int, default=None)
parser.add_argument('--HIDDEN_CHANNEL', type=int, default=None)
parser.add_argument('--DROPOUT', type=float, default=None)
parser.add_argument('--NLAYERS', type=int, default=None)
parser.add_argument('--HEADS_IN', type=int, default=None)
parser.add_argument('--HEADS_OUT', type=int, default=None)
parser.add_argument('--BATCH_SIZE', type=int, default=None)
parser.add_argument('--N_NEIGHBORS', type=int, default=None)

args = parser.parse_args()


def main(args):
    """Build yaml for wandb sweep

    Args:
        DATASET:        name of dataset to be used (pubmed, cora)
        METHOD:         sweep method (random, bayes, grid)
        TRAIN_FILE:     name of py file used for sweep
        YAML_FILE:      desired named of output yaml file
        IGNORE_KNOWN:   if True, only sweep params not reported in literature

    Returns:
        saved yaml file
    """
    assert args.METHOD.lower() in ['grid', 'random', 'bayes']
    assert args.DATASET.lower() in ['pubmed', 'cora', 'arxiv', 'products']
    assert args.TRAIN_FILE != None
    assert args.YAML_FILE != None

    # outline config dictionary
    sweep_config = {
        'program': args.TRAIN_FILE,
        'method': args.METHOD,
        'metric': {
            'goal': 'minimize',
            'name': 'epoch-val_loss'
        },
    }

    # add parameters
    # q_uniform in SIGN: https://arxiv.org/pdf/2004.11198v2.pdf
    param_dict = {
        'dataset': {
            'distribution': 'constant',
            'value': args.DATASET.lower()
        },
        'num_workers': {
            'distribution': 'constant',
            'value': args.N_WORKERS
        },
        'seed': {
            'distribution': 'constant',
            'value': 42
        },
        'optimizer_lr': {
            'distribution': 'uniform',
            'min': 1e-5,
            'max': 1e-1,
        },
        'optimizer_decay': {
            'distribution': 'uniform',
            'min': 1e-5,
            'max': 1e-1,
        },
        'epochs': {
            'distribution': 'int_uniform',
            'min': 5,
            'max': 100,
        },
        'hidden_channel': {
            'distribution': 'uniform',
            'values': [2**x for x in range(3, 13)],
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.8,
        },
        'heads_in': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 9,
        },
        'heads_out': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 9,
        },
        'nlayers': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 4,
        },
        'batch_size': {
            'distribution': 'q_uniform',
            'distribution': 'uniform',
            'values': [2**x for x in range(3, 13)],
        },
        'n_neighbors': {
            'values': [-1, 10, 20, 50, 60, 100, 150, 300, 500]
        }
    }

    sweep_config['parameters'] = param_dict

    # user specified parameters
    for k, v in vars(args).items():
        if v is not None:
            sweep_config['parameters'][k.lower()] = {'value': v}

    # if trial, simplify computation
    if args.RUN_TRIAL:
        sweep_config['method'] = 'random'
        sweep_config['parameters']['optimizer_lr'] = {'value': 1e-3}
        sweep_config['parameters']['optimizer_decay'] = {'value': 1e-3}
        sweep_config['parameters']['epochs'] = {'value': 5}
        sweep_config['parameters']['hidden_channel'] = {'value': 32}
        sweep_config['parameters']['dropout'] = {'value': 0.6}
        sweep_config['parameters']['heads_in'] = {'value': 8}
        sweep_config['parameters']['heads_out'] = {'value': 1}
        sweep_config['parameters']['nlayers'] = {'value': 1}
        sweep_config['parameters']['batch_size'] = {
            'value': 256}  # for GAT_loader
        sweep_config['parameters']['n_neighbors'] = {'value': 20}

    # write config to yaml file
    with open(args.YAML_FILE, 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

    # print final dictionary
    print(yaml.dump(sweep_config))


if __name__ == "__main__":
    main(args)
