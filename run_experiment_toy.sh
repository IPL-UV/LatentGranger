#!/usr/bin/env bash


gammas=(0 0.01 0.1 0.5 1 2 5 10 100)
for gamma in ${gammas[@]}; do
python main.py -d toy -m vae -g ${gamma} --maxlag 5 
done
