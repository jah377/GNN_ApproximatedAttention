# %%
import os
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from ogb.nodeproppred import PygNodePropPredDataset


# TODO running generate_data.py doesn't actually create folders and data

def make_cora(folder_path):
    folder_path = osp.join(folder_path, 'cora')

    if not osp.exists(folder_path):
        os.makedirs(folder_path)

    for K in range(6):

        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
            T.SIGN(K)
        ])

        # download
        dataset = Planetoid(
            root=folder_path,
            name='Cora',
            transform=transform,
            split='full',
        )

        filename = osp.join(folder_path, f'cora_sign_k{K}.pth')
        torch.save(dataset, filename)


def make_pubmed(folder_path):
    folder_path = osp.join(folder_path, 'pubmed')

    if not osp.exists(folder_path):
        os.makedirs(folder_path)

    for K in range(6):

        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
            T.SIGN(K)
        ])

        # download
        dataset = Planetoid(
            root=folder_path,
            name='Pubmed',
            transform=transform,
            split='full',
        )

        filename = osp.join(folder_path, f'pubmed_sign_k{K}.pth')
        torch.save(dataset, filename)


def make_products(folder_path):

    folder_path = osp.join(folder_path, 'products')

    if not osp.exists(folder_path):
        os.makedirs(folder_path)

    for K in range(6):

        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
            T.SIGN(K)
        ])

        # download
        dataset = PygNodePropPredDataset(
            'ogbn-products',
            root=folder_path,
            transform=transform)

        filename = osp.join(folder_path, f'products_sign_k{K}.pth')
        torch.save(dataset, filename)
        del transform, dataset


def make_arxiv(folder_path):

    folder_path = osp.join(folder_path, 'arxiv')

    if not osp.exists(folder_path):
        os.makedirs(folder_path)

    for K in range(6):

        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
            T.SIGN(K)
        ])

        # download
        dataset = PygNodePropPredDataset(
            'ogbn-arxiv',
            root=folder_path,
            transform=transform)

        filename = osp.join(folder_path, f'arxiv_sign_k{K}.pth')
        torch.save(dataset, filename)
        del transform, dataset
# %%


if __name__ == '__main__':
    # create directory
    folder_path = osp.join(os.getcwd(), 'data')

    if not osp.exists(folder_path):
        os.makedirs(folder_path)
        print("Directory '% s' created" % folder_path)

    make_cora(folder_path)
    make_pubmed(folder_path)
    make_products(folder_path)
    make_arxiv(folder_path)
