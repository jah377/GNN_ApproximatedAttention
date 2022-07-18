# An Approximate Attention Filter For The Efficient Precomputation Of Scalable Graph Convolutions

This repository contains the code corresponding to the thesis analysis presented in "An Approximate Attention Filter For the Efficient Precomputation of Scalable Graph Convolutions" submitted in partial fulfillment for the degree of MSc. in Data Science at the University of Amsterdam. 

<blockquote class="imgur-embed-pub" lang="en" data-id="a/y9sOOOH" data-context="false" ><a href="//imgur.com/a/y9sOOOH"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## Requirements

Dependencies with python 3.9.5:
```
torch==1.11.0
torch_geometric==2.0.4
torch_scatter==2.0.9
torch_sparse==0.6.13
ogb==1.3.3
einops==0.4.1
PyYAML==6.0
```

## Setup

Installing python dependencies in a virtual environment

```bash
# Creation and mount virtual environment
python3 -m venv ./approx_attn
source ./approx_attn/bin/activate

# Install CPU or GPU dependencies 
pip install -r requirements_GPU.txt
```

## Training and Evaluation





