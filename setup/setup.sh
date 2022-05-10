#!/bin/bash

# download and install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh # install in /nfs/code/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
conda update -n base -c defaults conda -y # update conda

# ========================================================================

# create CPU virtual environment
python3 -m venv venvs/CPU_venv
source venvs/CPU_venv/bin/activate

venvs/CPU_venv/bin/python3 -m pip install --upgrade pip
venvs/CPU_venv/bin/python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
venvs/CPU_venv/bin/python3 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
venvs/CPU_venv/bin/python3 -m pip install linformer-pytorch einops ogb wandb PyYAML
venvs/CPU_venv/bin/python3 -m pip install --upgrade wandb
wandb login

pip freeze >> setup/requirements_CPU.txt

deactivate 

# ========================================================================

# create GPU virtual environment
python3 -m venv venvs/GPU_venv
source venvs/GPU_venv/bin/activate

venvs/GPU_venv/bin/python3 -m pip install --upgrade pip
venvs/GPU_venv/bin/python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
venvs/GPU_venv/bin/python3 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
venvs/GPU_venv/bin/python3 -m pip install linformer-pytorch einops ogb wandb PyYAML
venvs/GPU_venv/bin/python3 -m pip install --upgrade wandb
wandb login

pip freeze >> setup/requirements_GPU.txt

deactivate 
