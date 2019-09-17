#!/bin/bash

chmod +x Anaconda3-2019.07-Linux-x86_64.sh
./Anaconda3-2019.07-Linux-x86_64.sh -b

export PATH=/home/vagrant/anaconda3/bin:$PATH

conda update conda

conda create --name myenv python=3.7
source activate myenv
conda install -y pandas matplotlib
conda install -y -c conda-forge xgboost scikit-learn keras

wget https://www.dropbox.com/s/3kg71wbhn7cic2p/data.tar.gz
tar xfz data.tar.gz
rm data.tar.gz
wget https://www.dropbox.com/s/qapuek0vtm4z8wu/mini-kaggle.tar.gz
tar xfz mini-kaggle.tar.gz
rm mini-kaggle.tar.gz

# download the code
wget https://www.dropbox.com/s/c8yvzkoqukmmw1w/kaggle.tar.gz
tar xfz kaggle.tar.gz
rm kaggle.tar.gz
cd kaggle
rm data sosc_data
ln -s ../data data
ln -s ../mini-kaggle mini-kaggle