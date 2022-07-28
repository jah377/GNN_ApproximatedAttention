# An Approximate Attention Filter For The Efficient Precomputation Of Scalable Graph Convolutions
*Master Thesis (5294MTD18Y)*

![ScreenShot](https://raw.github.com/jah377/ApproximatedAttention/main/report/github_image.png)

This repository contains code corresponding to the analyses presented in [An Approximate Attention Filter For the Efficient Precomputation of Scalable Graph Convolutions](https://github.com/jah377/ApproximatedAttention/blob/main/report/manuscript.pdf), submitted in partial fulfillment for the degree of MSc. in Data Science at the University of Amsterdam. 

## Motivation

Graph neural networks (GNNs) describe neural architectures generalized to learn on relational data. The structure intrinsic to social networks or e-commerce makes GNNs an attractive tool in industrial settings, however scalability presents a significant challenge. Newer decoupled models like Scalable Inception Graph Network (SIGN) improve scalability by approximating graph convolutional layers independently from parameter learning. As such, layers are replaced with linear diffusion operations of various filter sizes, which are efficiently pre-aggregated before training. However, diffusion is dependent on topology and does not consider node features, potentially limiting accuracy. The present study proposes enhancing diffusion in SIGN with a novel attention filter, approximating learned attention by characterizing feature-similarity between nodes. Experiments assessed the efficacy and efficiency of filters composed of learned graph attention weights, cosine similarity (CS), or a modified scaled dot product attention (DPA) on three medium- and large-scale benchmark datasets. See [results](#results).


## Requirements

Dependencies with python 3.9.7:
```
torch==1.11.0
torch_geometric==2.0.4
torch_scatter==2.0.9
torch_sparse==0.6.13
ogb==1.3.3
einops==0.4.1
```

## Setup

Installing python dependencies in a virtual environment

```bash
# Creation and mount virtual environment
python3 -m venv ./approx_attn
source ./approx_attn/bin/activate

# Install CPU or GPU dependencies 
pip install -r setup/requirements_GPU.txt
```

## Examples

### Baseline SIGN model
`Verbose=1` returns epoch-level results for every `EVAL_EVERY`-th epoch.

```bash
python main.py --VERBOSE 0 --DATASET 'pubmed' --ATTN_FILTER 'cosine' --EPOCHS 10 --EVAL_EVERY 5 --N_RUNS 3 --HOPS 2 --BATCH_SIZE 256 --LEARNING_RATE 1e-3 --WEIGHT_DECAY 1e-7 --INCEPTION_LAYERS 2 --INCEPTION_UNITS 512 --CLASSIFICATION_LAYERS 3 --CLASSIFICATION_UNITS 512 --FEATURE_DROPOUT 0.3 --NODE_DROPOUT 0.3 --BATCH_NORMALIZATION 1
```


### SIGN with Cosine Similarity Filter
`FILTER_BATCH_SIZE` determines the batch size when computing the dot product, and is limited by GPU memory. 

```bash
python main.py --VERBOSE 0 --DATASET 'pubmed' --ATTN_FILTER 'cosine' --EPOCHS 4 --EVAL_EVERY 2 --N_RUNS 1 --HOPS 3 --BATCH_SIZE 512 --LEARNING_RATE 1e-3 --WEIGHT_DECAY 1e-7 --INCEPTION_LAYERS 3 --INCEPTION_UNITS 256 --CLASSIFICATION_LAYERS 2 --CLASSIFICATION_UNITS 512 --FEATURE_DROPOUT 0.3 --NODE_DROPOUT 0.2 --BATCH_NORMALIZATION 1 --FILTER_BATCH_SIZE 100000 --ATTN_NORMALIZATION 1
```

### SIGN with Multi-head Dot Product Filter
`ATTN_HEADS=3` computes the similarity score averaged across all heads. 
```bash
python main.py --VERBOSE 1 --DATASET 'pubmed' --ATTN_FILTER 'dotprod' --EPOCHS 4 --EVAL_EVERY 2 --N_RUNS 1 --HOPS 3 --BATCH_SIZE 512 --LEARNING_RATE 1e-3 --WEIGHT_DECAY 1e-7 --INCEPTION_LAYERS 3 --INCEPTION_UNITS 256 --CLASSIFICATION_LAYERS 2 --CLASSIFICATION_UNITS 512 --FEATURE_DROPOUT 0.3 --NODE_DROPOUT 0.2 --BATCH_NORMALIZATION 1 --FILTER_BATCH_SIZE 100000 --ATTN_HEADS 3 --ATTN_NORMALIZATION 1
```

### SIGN with GAT Filter
*Must specify GAT model parameters*
```bash
python main.py --VERBOSE 1 --DATASET 'pubmed' --ATTN_FILTER 'gat' --EPOCHS 4 --EVAL_EVERY 2 --N_RUNS 1 --HOPS 5 --BATCH_SIZE 64 --LEARNING_RATE 1e-3 --WEIGHT_DECAY 1e-6 --INCEPTION_LAYERS 2 --INCEPTION_UNITS 64 --CLASSIFICATION_LAYERS 3 --CLASSIFICATION_UNITS 512 --FEATURE_DROPOUT 0.0 --NODE_DROPOUT 0.4 --BATCH_NORMALIZATION 1 --GAT_EPOCHS 10 --GAT_BATCH_SIZE 1024 --GAT_LEARNING_RATE 0.01 --GAT_WEIGHT_DECAY 1e-3 --GAT_HIDDEN_UNITS 8 --GAT_NODE_DROPOUT 0.6 --GAT_LAYERS 2 --GAT_HEADS_IN 8 --GAT_HEADS_OUT 8 --GAT_NEIGHBORS 150 --GAT_LR_PATIENCE 5
```


## Results
### Average Accuracy

| Model         | Pubmed        | ogbn-arXiv    | ogbn-products |
| ------------- | ------------- | ------------- | ------------- |
| GAT | 84.6 | OOM | OOM |
| SIGN | 89.5 | 54.7 | 75.5 |
| SIGN+GAT | 89.1 | OOM | OOM |
| SIGN+CS | 90.1 | 59.6 | 76.3 |
| SIGN+SHA | 90.1 | 59.1 | 77.3 |
| SIGN+MHA | 90.4 | 57.1 | 76.3 |

### Average Preprocessing Time
| Model         | Pubmed        | ogbn-arXiv    | ogbn-products |
| ------------- | ------------- | ------------- | ------------- |
| SIGN | 0.11 | 0.86 | 109.76 |
| SIGN+GAT | 356.66 | OOM | OOM |
| SIGN+CS | 0.32 | 2.51 | 299.74 |
| SIGN+SHA | 0.54 | 1.90 | 214.39 |
| SIGN+MHA | 2.16 | 6.03 | 559.47 |
