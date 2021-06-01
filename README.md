# LatentGranger 

## requirements 

- [`conda_requirements.txt`](conda_requirements.txt)
- [`pip_requirements.txt`](pip_requirements.txt)

## usage 

### train the autoencoder 

```
## this will train the autoencoder over the Toy dataset with beta = 0.01 nad max lag = 5
python3 main.py -d Toy -b 0.01 -l 5 
```

### XAI

The following command will extract the latent representation, average absolute gradients and 
neural integrated gradients. The output files will be available in the `viz/` folder. 
By default the last checkpoint for the last trained model is used but a specific trained model can 
be specified with `-t` (timepoint) and `-c` (checkpoint name).  

```
python3 explain.py -d Toy --extract --nig --grad --save
```
