# LatentGranger 

## requirements 

The `LatentGranger` code is developed with: 

- Python 3.9.7 
- pytorch 1.9.1  
- pytorch-lightning 1.4.7


The used conda (tested with v4.10.1) environment with the complete list of libraries 
 is  described in `environment.yaml`.  


## usage 

### train the autoencoder 

```
## this will train the autoencoder over the Toy dataset with gamma = 0.01 nad max lag = 5
python3 main.py -d toy -g 0.01 --maxlag 5 
```

### XAI

The following command will extract the latent representation, average absolute gradients and 
neural integrated gradients. The output files will be available in the `viz/` folder. 
By default the last checkpoint for the last trained model is used but a specific trained model can 
be specified with `-t` (timepoint) and `-c` (checkpoint name).  

```
python3 explain.py -d toy --extract --nig --grad --save
```
