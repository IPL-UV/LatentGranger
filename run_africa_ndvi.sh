#!/usr/bin/env bash

gammas=(40000 50000 60000 70000 80000 90000 100000)
for gamma in ${gammas[@]}; do
python main.py --arch vae -d africandvi -g ${gamma} --maxlag 10 --max_epochs 200 --limit_train_batches 5
done
