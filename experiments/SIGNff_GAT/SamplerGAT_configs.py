"""
Hyperparam sweep of GAT model performed separately, prior to SIGN
- params_dict contains best-performing hyperparam configuration
- manually entered in current file

**Theoretically possible to sweep GAT, return params, and run SIGN end-to-end
"""

# pubmed: https://arxiv.org/pdf/1710.10903.pdf
# cora: https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial3/Tutorial3.ipynb?pli=1#scrollTo=FyTEd5HK8_hQ

GATparams = {
    'cora': {
        'optimizer_type': 'Adam',
        'optimizer_lr': 0.005,
        'optimizer_decay': 0.0005,
        'epochs': 100,
        'hidden_channel': 8,
        'dropout': 0.6,
        'nlayers': 2,
        'heads_in': 8,
        'heads_out': 1,
        'batch_size': 512,  ## not tested
        'n_neighbors': 100, ## not tested 
    },
    'pubmed': {
        'optimizer_type': 'Adam',
        'optimizer_lr': 0.01,
        'optimizer_decay': 0.001,
        'epochs': 100,
        'hidden_channel': 8,
        'dropout': 0.6,
        'nlayers': 2,
        'heads_in': 8,
        'heads_out': 8,
        'batch_size': 1789,
        'n_neighbors': 150,
    },
}
