import yaml
import argparse

# parser.add_arg doesn't play nice with bool input
from distutils.util import strtobool


parser = argparse.ArgumentParser(description='inputs')

# general input arguments
parser.add_argument('--DATASET', type=str, default='cora')
parser.add_argument('--METHOD', type=str, default='random')
parser.add_argument('--TRAIN_FILE', type=str, default=None)
parser.add_argument('--YAML_FILE', type=str, default=None)
parser.add_argument('--IGNORE_KNOWN', type=strtobool, default=False)

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
            'distribution': 'q_uniform',
            'min': 8,
            'max': 1024,
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
            'min': 8,
            'max': 2048,
        },
        'n_neighbors': {
            'values': [-1, 10, 20, 50, 60, 100, 150, 300]
        }
    }

    sweep_config['parameters'] = param_dict

    # parameters already determined
    if args.IGNORE_KNOWN == True:
        if args.DATASET.lower() == 'pubmed':
            sweep_config['method'] = 'random'
            sweep_config['parameters']['optimizer_lr'] = {'value': 0.01}
            sweep_config['parameters']['optimizer_decay'] = {'value': 0.001}
            sweep_config['parameters']['epochs'] = {'value': 100}
            sweep_config['parameters']['hidden_channel'] = {'value': 8}
            sweep_config['parameters']['dropout'] = {'value': 0.6}
            sweep_config['parameters']['heads_in'] = {'value': 8}
            sweep_config['parameters']['heads_out'] = {'value': 8}
            sweep_config['parameters']['nlayers'] = {'value': 2}

    # write config to yaml file
    with open(args.YAML_FILE, 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

    # print final dictionary
    print(yaml.dump(sweep_config))


if __name__ == "__main__":
    main(args)
