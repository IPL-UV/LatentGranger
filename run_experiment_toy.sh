#!/usr/bin/env bash

arch=$1

gammas=(0 0.01 0.1 0.5 1 1.5 2 5 10 20 100)
for gamma in ${gammas[@]}; do
python main.py --arch ${arch} -d toy -g ${gamma} --maxlag 5 --max_epochs 100 --limit_train_batches 20 --earlystop --dir "toy_exp"
done
