#!/bin/bash

# download and install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh # install in /nfs/code/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
conda update -n base -c defaults conda -y # update conda

# export LD_LIBRARY_PATH="/home/jharris/miniconda3/lib:$LD_LIBRARY_PATH"

# create virtual environment -- conda env didn't work due to shell
python3 -m venv pyg-venv
source pyg-venv/bin/activate

# install dependencies
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-geometric 
pip install linformer-pytorch 
pip install einops
pip install ogb
pip install wandb
pip install PyYAML
pip install --upgrade wandb

# log in to W&B
wandb login

