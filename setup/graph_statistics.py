import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('--data_name', type=str, default='pubmed')
args = parser.parse_args()


def download_data(dataset_str):

    transform = T.Compose([T.NormalizeFeatures()])

    if dataset_str.lower() in ['products', 'arxiv']:
        dataset = PygNodePropPredDataset(
            f'ogbn-{dataset_str.lower()}',
            root=f'/tmp/{dataset_str.title}',
            transform=transform)
    else:
        dataset = Planetoid(
            root=f'/tmp/{dataset_str.title()}',
            name=f'{dataset_str.title()}',
            transform=transform,
            split='full',
        )

    return dataset


def standardize_data(data_obj, data_str):
    assert data_str.lower() in ['cora', 'pubmed', 'products', 'arxiv']

    # extract relevant information
    data = data_obj[0]
    data.num_classes = data_obj.num_classes
    data.num_nodes = data.num_nodes
    data.num_edges = data.num_edges
    data.num_node_features = data.num_node_features
    data.n_id = torch.arange(data.num_nodes)  # global node id

    # standardize mask -- node idx, not bool mask
    if data_str.lower() in ['products', 'arxiv']:
        masks = data_obj.get_idx_split()
        data.train_mask = masks['train']
        data.val_mask = masks['valid']
        data.test_mask = masks['test']
    else:
        data.train_mask = torch.where(data.train_mask)[0]
        data.val_mask = torch.where(data.val_mask)[0]
        data.test_mask = torch.where(data.test_mask)[0]

    return data


def percent_split(data):
    n_train = data.train_mask.shape[0]
    n_val = data.val_mask.shape[0]
    n_test = data.test_mask.shape[0]
    n_total = n_train+n_val+n_test

    return {
        'train': n_train/n_total,
        'val': n_val/n_total,
        'test': n_test/n_total,
    }


def total_edges(data):
    return data.num_edges if data.is_directed() else data.num_edges/2


def average_degree(data):
    return data.num_edges/data.num_nodes


def homophily_degree(data):
    r, c = data.edge_index
    y = data.y
    return sum(y[r] == y[c])/len(r)


def get_statistics(args):
    data_name = args.data_name

    # data = download_data(data_name)
    data = torch.load(f'data/{data_name}/{data_name}_sign_k0.pth')

    data = standardize_data(data, data_name)
    splits = percent_split(data)
    n_edges = total_edges(data)
    avg_deg = average_degree(data)
    homo_deg = homophily_degree(data)

    print()
    print(f'\n==== {data_name.upper()} Statistics =====')
    print('Nodes: {:,}'.format(data.num_nodes))
    print('Edges: {:,}'.format(n_edges))
    print('Features: {:,}'.format(data.num_features))
    print('Classes: {:,}'.format(data.num_classes))
    print('Directed?: {:}'.format(data.is_directed()))
    print('%Train: {}'.format(round(splits['train'], 3)))
    print('%Val: {}'.format(round(splits['val'], 3)))
    print('%Test: {}'.format(round(splits['test'], 3)))
    print('Avg. Deg.: {}'.format(round(avg_deg, 3)))
    print('Homophily Deg.: {}'.format(round(homo_deg.item(), 3)))
    print()

    del data


if __name__ == '__main__':
    get_statistics(args)
