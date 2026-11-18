#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n crimson python=3.12 -y
conda activate crimson
pip install --upgrade pip
cd "$(dirname "$0")/../CRIMSON"
pip install -e .
