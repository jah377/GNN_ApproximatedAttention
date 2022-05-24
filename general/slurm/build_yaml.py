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

    GAT_models = ['fullbatchgat', 'samplergat']
    SIGN_models = ['sign', 'sign_fullbatchgat',
                   'sign_samplergat', 'sign_sha', 'sign_mha']
    DPA_models = ['sign_sha', 'sign_mha']
    CS_models = ['sign_cs']

    include_gat_params = args.MODEL.lower() in GAT_models
    include_sign_params = args.MODEL.lower() in SIGN_models
    include_dpa_params = args.MODEL.lower() in DPA_models
    include_cs_params = args.MODEL.lower() in CS_models

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
    }

    sweep_config['parameters'] = param_dict

    # add model-specific parameters
    if include_sign_params:
        param_dict.update({
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
        })

        if args.MODEL.lower() == 'sign_samplergat':
            param_dict.update({
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
            })

    if include_gat_params:
        if args.MODEL.lower() == 'samplergat':
            param_dict.update({
                'batch_size': {
                    'distribution': 'q_uniform',
                    'min': 8,
                    'max': 2048,
                },
            })

        param_dict.update({
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
        })

    if include_dpa_params:
        if args.MODEL.lower() == 'sign_mha':
            param_dict.update({
                'attn_heads': {
                    'distribution': 'int_uniform',
                    'min': 1,
                    'max': 5,
                },
            })
        elif args.MODEL.lower() == 'sign_sha':
            param_dict.update({
                'attn_heads': {
                    'distribution': 'constant',
                    'value': 1
                },
            })

    if include_cs_params:
        param_dict.update({
            'cs_batch_size': {
                'distribution': 'constant',
                'value': 50000
            },
        })

    # reduce complexity for trialing
    if args.RUN_TRIAL:
        sweep_config['method'] = 'random'
        sweep_config['parameters']['optimizer_lr'] = {'value': 1e-3}
        sweep_config['parameters']['optimizer_decay'] = {'value': 1e-3}
        sweep_config['parameters']['epochs'] = {'value': 5}
        sweep_config['parameters']['hidden_channel'] = {'value': 32}
        sweep_config['parameters']['dropout'] = {'value': 0.6}

        if include_gat_params:
            sweep_config['parameters']['heads_in'] = {'value': 8}
            sweep_config['parameters']['heads_out'] = {'value': 1}
            sweep_config['parameters']['nlayers'] = {'value': 1}
            sweep_config['parameters']['batch_size'] = {
                'value': 256}  # for GAT_loader

        if include_sign_params:
            sweep_config['parameters']['K'] = {'value': 1}
            sweep_config['parameters']['batch_norm'] = {'value': 1}
            sweep_config['parameters']['batch_size'] = {'value': 256}

    # write config to yaml file
    with open(args.YAML_FILE, 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

    # print final dictionary
    print(yaml.dump(sweep_config))


if __name__ == "__main__":
    main(args)
