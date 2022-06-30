import yaml
import argparse

# parser.add_arg doesn't play nice with bool input
from distutils.util import strtobool


parser = argparse.ArgumentParser(description='inputs')

# general input arguments
parser.add_argument('--DATASET', type=str, default='products',
                    help='name of dataset')
parser.add_argument('--MODEL', type=str, default=None, help='name of model')
parser.add_argument('--METHOD', type=str, default='random',
                    help='grid, bayes, random')
parser.add_argument('--TRAIN_FILE', type=str,
                    default=None, help='sweep.py file')
parser.add_argument('--YAML_FILE', type=str, default=None,
                    help='name of output yaml file')
parser.add_argument('--RUN_TRIAL', type=strtobool,
                    default=True, help='reduce complexity for testing')

parser.add_argument('--SEED', type=int, default=None, help='seed value')
parser.add_argument('--EPOCHS', type=int, default=None,
                    help='number of epochs')
parser.add_argument('--HOPS', type=int, default=None,
                    help='k-hop neighborhood aggregations')
parser.add_argument('--BATCH_SIZE', type=int, default=None,
                    help='DataLoader batch size')
parser.add_argument('--LEARNING_RATE', type=float,
                    default=None, help='optimizer learning rate')
parser.add_argument('--WEIGHT_DECAY', type=float, default=None,
                    help='optimizer regularization param')
parser.add_argument('--INCEPTION_LAYERS', type=int, default=None,
                    help='number of inception feed-forward layers')
parser.add_argument('--INCEPTION_UNITS', type=int,
                    default=None, help='inception hidden channel size')
parser.add_argument('--CLASSIFICATION_LAYERS', type=int, default=None,
                    help='number of classification feed-forward layers ')
parser.add_argument('--CLASSIFICATION_UNITS', type=int,
                    default=None, help='classification hidden channel size')
parser.add_argument('--FEATURE_DROPOUT', type=float,
                    default=None, help='fraction of features to be dropped')
parser.add_argument('--NODE_DROPOUT', type=float, default=None,
                    help='fraction of NN nodes to be dropped')
parser.add_argument('--BATCH_NORMALIZATION', type=strtobool,
                    default=None, help='NN regularization')
parser.add_argument('--LR_PATIENCE', type=int, default=None,
                    help='update scheduler LR after n epochs')
parser.add_argument('--TERMINATION_PATIENCE', type=int, default=None,
                    help='terminate sweep after n epochs w/o val_loss improvement ')
parser.add_argument('--TRANSFORMATION', type=int, default=None,
                    help="'cosine', 'gat','dot_product', 'cosine_per_k' ")
# parser.add_argument('--ATTN_HEADS', type=int, default=None, help='number of attention heads (DPA only)')
# parser.add_argument('--DPA_NORMALIZATION', type=strtobool, default=None, help='row min-max normalization of DPA weights (DPA only)')
parser.add_argument('--CS_BATCH_SIZE', type=int, default=None,
                    help='batch size for CosineSimilarity calc. (CS only)')

args = parser.parse_args()


def main(args):
    """ save sweep yaml to file """
    assert args.METHOD.lower() in ['grid', 'random', 'bayes']
    assert args.DATASET.lower() in ['pubmed', 'cora', 'arxiv', 'products', 'arxiv_new']
    assert args.MODEL != None
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

    # initial parameter inputs
    param_dict = {
        'DATASET': {
            'value': args.DATASET.lower()
        },
        'SEED': {
            'value': 42
        },
        'EPOCHS': {
            'value': 300,
        },
        'HOPS': {
            'values': [x for x in range(6)]
        },
        'BATCH_SIZE': {
            'values': [2048, 4096, 8192, 16384]
        },
        'LEARNING_RATE': {
            'values': [eval(f'1e-{x}') for x in range(8)]
        },
        'WEIGHT_DECAY': {
            'values': [eval(f'1e-{x}') for x in range(8)]
        },
        'INCEPTION_LAYERS': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 3,
        },
        'INCEPTION_UNITS': {
            'values': [128, 256, 512, 1024]
        },
        'CLASSIFICATION_LAYERS': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 3,
        },
        'CLASSIFICATION_UNITS': {
            'values': [128, 256, 512, 1024]
        },
        'FEATURE_DROPOUT': {
            'values': [x/10 for x in range(0, 8)],
        },
        'NODE_DROPOUT': {
            'values': [x/10 for x in range(0, 8)],
        },
        'BATCH_NORMALIZATION': {
            'value': 1,
        },
        'LR_PATIENCE': {
            'value': 5,
        },
        'TERMINATION_PATIENCE': {
            'value': 10,
        },
        'TRANSFORMATION': {
            'value': 'cosine'
        },
        # 'ATTN_HEADS': {
        #     'distribution': 'int_uniform',
        #     'min': 1,
        #     'max': 5,
        # },
        'ATTN_NORMALIZATION': {
            'values': [0, 1],
        },
        'FILTER_BATCH_SIZE': {
            'values': [100000],
        },
    }

    sweep_config['parameters'] = param_dict

    # if Single-Head DPA -> number of heads = 1
    if 'sha' in args.MODEL.lower():
        sweep_config['parameters']['ATTN_HEADS'] = {'value': 1}

    # if smaller datasets -> reduce range of values
    if args.DATASET.lower() in ['cora', 'pubmed']:
        sweep_config['parameters']['BATCH_SIZE'] = {
            'values': [64, 128, 256, 512, 1024]}
        sweep_config['parameters']['INCEPTION_UNITS'] = {
            'values': [64, 128, 256, 512]}
        sweep_config['parameters']['CLASSIFICATION_UNITS'] = {
            'values': [64, 128, 256, 512]}

    if args.DATASET.lower() in ['products']:
        sweep_config['parameters']['ATTN_NORMALIZATION'] = {'value': 1}

    # if user-determined value -> update parameter
    non_params = ['DATASET', 'MODEL', 'METHOD',
                  'TRAIN_FILE', 'YAML_FILE', 'RUN_TRIAL']
    for k, v in vars(args).items():
        if (k not in non_params) & (v is not None):
            if not isinstance(v, list):
                v = list(v)
            sweep_config['parameters'][k] = {f'values': v}

    # if running trial -> reduce complexity
    if args.RUN_TRIAL:
        sweep_config['method'] = 'random'
        sweep_config['parameters']['LEARNING_RATE'] = {'value': 1e-3}
        sweep_config['parameters']['WEIGHT_DECAY'] = {'value': 1e-3}
        sweep_config['parameters']['EPOCHS'] = {'value': 5}
        sweep_config['parameters']['INCEPTION_UNITS'] = {'value': 2048}
        sweep_config['parameters']['CLASSIFICATION_UNITS'] = {'value': 2048}
        sweep_config['parameters']['INCEPTION_LAYERS'] = {'value': 1}
        sweep_config['parameters']['CLASSIFICATION_LAYERS'] = {'value': 1}
        sweep_config['parameters']['NODE_DROPOUT'] = {'value': 0.6}
        sweep_config['parameters']['FEATURE_DROPOUT'] = {'value': 0.2}
        sweep_config['parameters']['HOPS'] = {'value': 1}
        sweep_config['parameters']['BATCH_SIZE'] = {'value': 256}
        sweep_config['parameters']['LR_PATIENCE'] = {'value': 1}
        sweep_config['parameters']['TERMINATION_PATIENCE'] = {'value': 20}

    # write config to yaml file
    with open(args.YAML_FILE, 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

    # print final dictionary
    print(yaml.dump(sweep_config))


if __name__ == "__main__":
    main(args)
