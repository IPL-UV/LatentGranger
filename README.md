# LatentGranger 

## requirements 

The `LatentGranger` code is developed with: 

- Python 3.9.7 
- pytorch 1.9.1  
- pytorch-lightning 1.4.7

The used anaconda (tested with v4.10.1) environment with the complete list of
libraries is  described in `environment.yaml`.  

## data

The toy dataset can be generated with 

```
Rscript generate_toy.R 
```

## architectures

### beta vae with fully connected

### beta vae with convolutional layers

TO FIX

## usage 

### train the autoencoder 

```
## this will train a simple (Granger)-VAE with fully conected layers over the Toy dataset 
## with gamma = 100 nad max lag = 5
python3 main.py -d toy --arch vae -g 100 --maxlag 5  
```

### XAI

The following command will extract the latent representation, average absolute gradients, 
neural integrated gradients and latent interventions. The output files will be available in the `viz/` folder. 
By default the last checkpoint for the last trained model is used but a specific trained model can 
be specified with `-t` (timepoint) and `-c` (checkpoint name).  

```
python3 explain.py -d toy --extract --nig --grad --save
```
