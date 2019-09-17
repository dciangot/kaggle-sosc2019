#!/bin/bash

sudo apt install -y python3
curl https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_65.sh | sh -

conda update conda

conda create --name myenv python=3.7
conda install pandas matplotlib
conda install -c conda-forge xgboost scikit-learn keras
