#!/bin/bash

# ============================================
# Python Test Generator Setup + Run Script
# ============================================

set -e

ENV_NAME="testgen"

echo "============================================"
echo "Creating conda environment..."
echo "============================================"

conda create -y -n $ENV_NAME python=3.11

echo "============================================"
echo "Activating environment..."
echo "============================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "============================================"
echo "Installing PyTorch..."
echo "============================================"

pip install torch

echo "============================================"
echo "Installing Hugging Face packages..."
echo "============================================"

pip install -r requirements.txt

echo "============================================"
echo "Running training..."
echo "============================================"

mkdir -p outputs
python train.py | tee outputs/train_log.txt

echo "============================================"
echo "Running inference..."
echo "============================================"

python inference.py | tee outputs/inference_log.txt

echo "============================================"
echo "Done."
echo "============================================"