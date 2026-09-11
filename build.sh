#!/bin/bash

conda create -y --name omxpu python=3.10
conda activate omxpu

python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install -r requirements.txt