#!/usr/bin/env bash

gammas=(100000 10000 10000 10000)
for gamma in ${gammas[@]}; do
python main.py --arch vae -d africandvi -g ${gamma} --maxlag 10 --max_epochs 200 --limit_train_batches 5
done
