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
parser.add_argument('--SEED', type=int, default=None)
parser.add_argument('--OPTIMIZER_LR', type=float, default=None)
parser.add_argument('--OPTIMIZER_DECAY', type=float, default=None)
parser.add_argument('--EPOCHS', type=int, default=None)
parser.add_argument('--HIDDEN_CHANNEL', type=int, default=None)
parser.add_argument('--DROPOUT', type=float, default=None)
parser.add_argument('--K', type=int, default=None)
parser.add_argument('--BATCH_NORM', type=strtobool, default=None)
parser.add_argument('--BATCH_SIZE', type=int, default=None)
parser.add_argument('--ATTN_HEADS', type=int, default=None)


args = parser.parse_args()


def main(args):
    """Build yaml for wandb sweep

    Args:
        DATASET:        name of dataset to be used (pubmed, cora)
        MODEL:          name of model (GAT_fullbatch, SIGN, SGcomb, SGsep)
        METHOD:         sweep method (random, bayes, grid)
        TRAIN_FILE:     name of py file used for sweep
        YAML_FILE:      desired named of output yaml file
        RUN_TRIAL:      if True, reduce complexity for efficent test of script

    Returns:
        saved yaml file
    """
    assert args.METHOD.lower() in ['grid', 'random', 'bayes']
    assert args.DATASET.lower() in ['pubmed', 'cora', 'arxiv', 'products']
    assert args.MODEL != None
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

    # add parameters (not model specific) to config dictionary
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
        'K': {
            'distribution': 'int_uniform',
            'min': 0,
            'max': 5,
        },
        'batch_norm': {
            'distribution': 'int_uniform',
            'min': 0,
            'max': 1,
        },
        'batch_size': {
            'distribution': 'q_uniform',
            'min': 8,
            'max': 2048,
        },
        'attn_heads': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 6,
        }
    }

    sweep_config['parameters'] = param_dict

    # user specified parameters
    for k, v in vars(args).items():
        if v is not None:
            sweep_config['parameters'][k.lower()] = {'value': v}

    # reduce complexity for trialing
    if args.RUN_TRIAL:
        sweep_config['method'] = 'random'
        sweep_config['parameters']['optimizer_lr'] = {'value': 1e-3}
        sweep_config['parameters']['optimizer_decay'] = {'value': 1e-3}
        sweep_config['parameters']['epochs'] = {'value': 5}
        sweep_config['parameters']['hidden_channel'] = {'value': 32}
        sweep_config['parameters']['dropout'] = {'value': 0.6}
        sweep_config['parameters']['K'] = {'value': 1}
        sweep_config['parameters']['batch_norm'] = {'value': 1}
        sweep_config['parameters']['batch_size'] = {'value': 256}
        sweep_config['parameters']['attn_heads'] = {'value': 2}

    # write config to yaml file
    with open(args.YAML_FILE, 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

    # print final dictionary
    print(yaml.dump(sweep_config))


if __name__ == "__main__":
    main(args)
