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
            'name': 'val_loss'
        },
    }

    # add parameters
    # q_uniform in SIGN: https://arxiv.org/pdf/2004.11198v2.pdf
    param_dict = {
        'DATASET': {
            'value': args.DATASET.lower()
        },
        'NUM_WORKERS': {
            'value': args.N_WORKERS
        },
        'SEED': {
            'value': 42
        },
        'EPOCHS': {
            'value': 300,
        },
        'BATCH_SIZE': {
            'values': [1024, 2048, 4096, 8192, 16384]
        },
        'LEARNING_RATE': {
            'values': [eval(f'1e-{x}') for x in range(8)]
        },
        'WEIGHT_DECAY': {
            'values': [eval(f'1e-{x}') for x in range(8)]
        },
        'N_LAYERS': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 4,
        },
        'HEADS_IN': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5,
        },
        'HEADS_OUT': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5,
        },
        'N_NEIGHBORS': {
            'values': [-1, 10, 20, 50, 100, 150, 300, 500]
        },
        'HIDDEN_UNITS': {
            'values': [128, 256, 512, 1024]
        },
        'NODE_DROPOUT': {
            'values': [x/10 for x in range(0, 8)],
        },
        'LR_PATIENCE': {
            'value': 5,
        },
        'TERMINATION_PATIENCE': {
            'value': 10,
        },
    }

    sweep_config['parameters'] = param_dict

    # reduce batchsize for large datasets
    if args.DATASET.lower() != 'pubmed':
        sweep_config['parameters']['BATCH_SIZE'] = {
            'values': [64, 128, 256, 512, 1024]}
        sweep_config['parameters']['N_NEIGHBORS'] = {
            'values': [10, 20, 50, 60, 100, 150]}

    # if user-determined value -> update parameter
    non_params = ['DATASET', 'MODEL', 'METHOD',
                  'TRAIN_FILE', 'YAML_FILE', 'RUN_TRIAL']
    for k, v in vars(args).items():
        if (k not in non_params) & (v is not None):
            if not isinstance(v, list):
                v = list(v)
            sweep_config['parameters'][k] = {f'values': v}

    # if trial, simplify computation
    if args.RUN_TRIAL:
        sweep_config['method'] = 'random'
        sweep_config['parameters']['LEARNING_RATE'] = {'value': 1e-3}
        sweep_config['parameters']['WEIGHT_DECAY'] = {'value': 1e-3}
        sweep_config['parameters']['EPOCHS'] = {'value': 5}
        sweep_config['parameters']['HIDDEN_UNITS'] = {'value': 32}
        sweep_config['parameters']['NODE_DROPOUT'] = {'value': 0.6}
        sweep_config['parameters']['HEADS_IN'] = {'value': 8}
        sweep_config['parameters']['HEADS_OUT'] = {'value': 1}
        sweep_config['parameters']['N_LAYERS'] = {'value': 1}
        sweep_config['parameters']['BATCH_SIZE'] = {
            'value': 256}  # for GAT_loader
        sweep_config['parameters']['N_NEIGHBORS'] = {'value': 20}

    # write config to yaml file
    with open(args.YAML_FILE, 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

    # print final dictionary
    print(yaml.dump(sweep_config))


if __name__ == "__main__":
    main(args)
