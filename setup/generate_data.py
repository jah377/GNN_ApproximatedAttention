import os
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset


# TODO running generate_data.py doesn't actually create folders and data


def make_cora(folder_path):
    for K in range(6):
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.SIGN(K)
        ])

        dataset = Planetoid(
            root='/tmp/Cora',
            name='Cora',
            transform=transform,
            split='full',
            )
        data = dataset[0]

        filename = osp.join(folder_path, f'cora_sign_k{K}.pth')
        torch.save(data, filename)


def make_pubmed(folder_path):
    for K in range(6):
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.SIGN(K)
        ])

        dataset = Planetoid(
            root='/tmp/Pubmed',
            name='Pubmed',
            transform=transform,
            split='full',
            )
        data = dataset[0]

        filename = osp.join(folder_path, f'pubmed_sign_k{K}.pth')
        torch.save(data, filename)


def make_products(folder_path):
    for K in range(6):
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.SIGN(K)
        ])

        dataset = PygNodePropPredDataset(
            'ogbn-products', 
            root='/tmp/Products', 
            transform=transform)
        
        filename = osp.join(folder_path, f'products_sign_k{K}.pth')
        torch.save(dataset, filename)


if __name__ == '__main__':
    # create directory
    dirpath = osp.dirname(osp.realpath('generate_data.py'))
    folder_path = osp.join(dirpath, '..', 'data')

    if not osp.exists(folder_path):
        os.makedirs(folder_path)
        print("Directory '% s' created" % folder_path)

    make_cora(folder_path)
    make_pubmed(folder_path)
    make_products(folder_path)
