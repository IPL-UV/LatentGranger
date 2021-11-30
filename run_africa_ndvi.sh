#!/usr/bin/env bash

gammas=(0 5000 10000 20000 21000 22000 23000 24000 25000 30000 35000 40000 50000 60000 70000)
for gamma in ${gammas[@]}; do
python main.py --arch vae -d africandvi -g ${gamma} --maxlag 10 --max_epochs 200 --limit_train_batches 5
done
