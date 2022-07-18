import os
import argparse
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset

parser = argparse.ArgumentParser(description='SIGN')
parser.add_argument('--PATH', type=str, default=None,
                    help='directory to store data/args.DATASET')
parser.add_argument('--DATASET', type=str, default=None,
                    help='name of dataset to be downloaded')
parser.add_argument('--HOPS', type=int, default=5, help='number of k-hops')
args = parser.parse_args()


def prep_data(path: str, dataset_name: str, K: int):
    """ standardize format of data object """
    possible_datasets = ['cora', 'pubmed', 'products', 'arxiv']
    dataset_name = dataset_name.lower()
    assert dataset_name in possible_datasets, f'Dataset {dataset_name} not available'

    # download data
    if dataset_name == 'arixv':
        transform = T.Compose([
            T.ToUndirected(),
            T.AddSelfLoops(),
            T.SIGN(K)
        ])
    else:
        transform = T.Compose([
            T.SIGN(K)
        ])

    if dataset_name in ['arxiv', 'products']:
        dataset = PygNodePropPredDataset(
            f'ogbn-{dataset_name}',
            root=path,
            transform=transform
        )
    else:
        dataset = Planetoid(
            root=path,
            name=dataset_name.title(),
            transform=transform,
            split='full'
        )

    # extract relevant information
    data = dataset[0]
    data.dataset_name = dataset_name.lower()
    data.num_nodes, data.num_feats = data.x.shape
    data.num_classes = dataset.num_classes
    data.n_id = torch.arange(data.num_nodes)  # global node id

    # standardize idx max
    if hasattr(dataset, 'get_idx_split'):
        masks = dataset.get_idx_split()
        data.train_mask = masks['train']
        data.val_mask = masks['valid']
        data.test_mask = masks['test']
        data.y = data.y.flatten()
    else:
        data.train_mask = torch.where(data.train_mask)[0]
        data.val_mask = torch.where(data.val_mask)[0]
        data.test_mask = torch.where(data.test_mask)[0]
    return data


if __name__ == '__main__':
    assert args.DATASET is not None, f'Dataset: Please enter dataset'
    assert args.DATASET in ['cora', 'pubmed', 'products',
                            'arxiv'], f'Dataset: {args.DATASET} not available'
    dataset_name = args.DATASET.lower()

    dir_path = os.getcwd() if args.PATH is None else args.PATH
    folder_path = osp.join(os.getcwd(), 'data')
    dataset_path = osp.join(folder_path, dataset_name)

    if not osp.exists(folder_path):
        os.makedirs(folder_path)

    if not osp.exists(dataset_path):
        os.makedirs(dataset_path)

    data = prep_data(dataset_path, dataset_name, args.HOPS)

    for i in range(args.HOPS, -1, -1):
        filename = osp.join(dataset_path, f'{dataset_name}_sign_k{i}.pth')
        torch.save(data, filename)
        print(f'COMPLETED: {filename}')
        del data[f'x{i}']
    del data
