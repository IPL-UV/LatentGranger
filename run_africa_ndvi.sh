#!/usr/bin/env bash

gammas=(0 10000 10000 20000 20000)
for gamma in ${gammas[@]}; do
python main.py --arch vae -d africandvilowres -g ${gamma} --maxlag 10 --max_epochs 400 --limit_train_batches 5
done
