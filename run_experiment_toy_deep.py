#!/usr/bin/env bash


gammas=(0 0.01 0.1 0.5 1 2 5 10 20 100)
for gamma in ${gammas[@]}; do
python main.py -d toy -m vaedeep -g ${gamma} --maxlag 5 --epoch 300 
done
