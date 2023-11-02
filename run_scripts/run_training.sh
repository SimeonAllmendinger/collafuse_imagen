#!/bin/bash

#
if [ ! -d "$TMP/distributedgenai/" ]; then
    mkdir $TMP/distributedgenai/
fi

if [ ! -d "$TMP/distributedgenai/data/" ]; then
    mkdir $TMP/distributedgenai/data/
fi

if [ ! -d "$TMP/distributedgenai/src/" ]; then
    mkdir $TMP/distributedgenai/src/
fi

# Extract compressed input data files on local SSD
## Data
if [ ! -d "$TMP/distributedgenai/data/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/distributedgenai/data.tgz
fi

## Assets
if [ ! -d "$TMP/distributedgenai/src/assets/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/distributedgenai/src/assets.tgz
fi

if [ ! -d "$TMP/distributedgenai/src/components/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/distributedgenai/src/components.tgz
fi

## Configs
if [ ! -d "$TMP/distributedgenai/configs/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/distributedgenai/configs.tgz
fi

## Scripts
if [ ! -d "$TMP/distributedgenai/run_scripts/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/distributedgenai/run_scripts.tgz
fi

## Virtual Environment
if [ ! -d "$TMP/distributedgenai/.venv/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/distributedgenai/.venv.tgz
fi

# Create results and scripts directory
mkdir $TMP/distributedgenai/run_scripts
mkdir $TMP/distributedgenai/results
cd $TMP/distributedgenai/results

# Activate virtual environment (venv)
cd $TMP/distributedgenai/
source ./.venv/bin/activate

# Start parameter tuning
./.venv/bin/python3 ./src/components/training.py --path_data_dir=$TMP/distributedgenai/
#./venv/bin/python3 ./src/components/dalle2/model/build_dalle2.py --path_data_dir=$TMP/distributedgenai/

cp -r $TMP/distributedgenai/src/assets/elucidated_imagen/models $HOME/distributedgenai/src/assets/elucidated_imagen/
