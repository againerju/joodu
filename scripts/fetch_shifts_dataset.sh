#!/bin/bash

# Folder structure
mkdir -p data/Shifts
cd data/Shifts

# Download train, dev and eval set
wget https://storage.yandexcloud.net/yandex-research/shifts/sdc/canonical-trn-dev-data.tar
wget https://storage.yandexcloud.net/yandex-research/shifts/sdc/canonical-eval-data.tar

# Extract
tar xvf canonical-eval-data.tar 
tar xvf canonical-eval-data/evaluation.tar.gz
tar xvf canonical-trn-dev-data.tar
tar xvf canonical-trn-dev-data/development.tar.gz
tar xvf canonical-trn-dev-data/train.tar.gz
