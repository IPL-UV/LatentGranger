#!/usr/bin/env bash

gammas=(0 5000 7500 10000)
for gamma in ${gammas[@]}; do
python main.py --arch vae -d africandvi -g ${gamma} --maxlag 5 --max_epochs 200 --limit_train_batches 5 --dir "exp_africa_ndvi"
done
