#!/usr/bin/env bash

gammas=(10000 10000 10000)
lags=(5 10 20)
for gamma in ${gammas[@]}; do
for lag in ${lags[@]}; do
python main.py --arch vae -d ndvilowres -g ${gamma} --maxlag ${lag} --max_epochs 200 --limit_train_batches 5
done
done
