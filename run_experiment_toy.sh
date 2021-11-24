#!/usr/bin/env bash

gammas=(0 10 100 250 500 750 1000 2500 5000)
for gamma in ${gammas[@]}; do
python main.py --arch vaetoy -d toy -g ${gamma} --maxlag 5 --max_epochs 200 --limit_train_batches 10 --dir "toy_exp"
done
