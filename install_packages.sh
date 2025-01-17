#!/bin/bash
pip install pytorch_lightning==1.7.7
pip install -e .
pip install scikit-survival==0.19.0
pip install pycox
pip install lifelines
pip install pyro-ppl
pip uninstall horovod -y
HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod
pip install pysurvival
pip install wandb
pip install pandas==1.5
pip install numpy==1.20.3
