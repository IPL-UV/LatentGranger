#!/usr/bin/env bash

gammas=(0 1000 2500 5000 7500 10000 20000)
for gamma in ${gammas[@]}; do
python main.py --arch vae -d africandvilowres -g ${gamma} --maxlag 10 --max_epochs 200 --limit_train_batches 5
done
