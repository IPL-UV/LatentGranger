#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explain latent space of LatentGranger


Anonymized code submitted alongide 
the manuscript titled 
Learning Granger Causal Feature Representations 

please do not distribute
"""

import os
import numpy as np
import argparse, yaml
from datetime import datetime
from shutil import copyfile


import netCDF4 as nc
from natsort import natsorted
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image

# Model
import model 
from model import lag_cor  
from model import granger_loss 
from utils import *

# PyTorch Captum for XAI
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance, LayerIntegratedGradients
from captum.attr import NeuronConductance, NeuronIntegratedGradients


# ArgParse
parser = argparse.ArgumentParser(description="ArgParse")
parser.add_argument('-a', '--arch', default='LatentGranger', type=str,
                  help='arch name (default: NeuralGranger)')
parser.add_argument('-d', '--database', default='toy', type=str,
                  help='database name (default: toy)')
parser.add_argument('-c','--checkpoint', default='last.ckpt', type=str,
                  help='checkpoint (default: last)')
parser.add_argument('-t','--timestamp', default='', type=str,
                  help='timestampt (default: '')')
parser.add_argument('--train', action = 'store_true',
                  help='use trainig data')
parser.add_argument('--val', action = 'store_true',
                  help='use val data')
parser.add_argument('--save', action = 'store_true',
                  help='save image')
parser.add_argument('--fast', action = 'store_true',
                  help='only fast part')
parser.add_argument('--grad', action = 'store_true',
                  help='only fast part')
parser.add_argument('--extract', action = 'store_true',
                  help='extract on LC')
parser.add_argument('--nig', action = 'store_true',
                  help='run NIG')


args = parser.parse_args()

chckroot = os.path.join('checkpoints', args.database, args.arch)
logsroot = os.path.join('logs', args.database, args.arch, args.arch)

allchckpts = natsorted(
            [
                fname
                for fname in os.listdir(chckroot)
            ],
        )

if args.timestamp == '':
    chosen = allchckpts[-1]
else:
    ### search the closer one 
    ### .... to do ... 
    #for timestamp in timestamps:
    dt = datetime.fromisoformat(args.timestamp) 
    chosen = min(allchckpts,key=lambda x : abs(datetime.fromisoformat(x) - dt))

checkpoint = os.path.join(chckroot, chosen, args.checkpoint)

print('chosen checkpoint: ' + checkpoint)

## define the model and load the checkpoint
model = getattr(model, args.arch).load_from_checkpoint(checkpoint)

print("model loaded with chosen checkpoint") 

################### print model info #####################

print(model)
print(f'beta: {model.beta}, lag: {model.lag}')

#####################   here we can do inference, plot ##############


if args.train:
    data = model.data_train
elif args.val:
    data = model.data_val
else:
    data = model.data_test

savedir = os.path.join('viz', chosen)
os.makedirs(savedir, exist_ok = True)

x, target = data[0] 
x = torch.reshape(x, (1,) + x.shape)
x.requires_grad_()
target = torch.reshape(target, (1,) + target.shape)

model.eval()
x_out, latent, mu, sigma = model(x)


for j in range(latent.shape[-1]):
    corr = lag_cor(latent[:,:,j], target, lag = model.lag) 
    print(f'lagged correlation with target of {j}th latent: {corr}')

gloss = granger_loss(latent, target, maxlag = model.lag)

print(f'granger losss: {gloss}')


if args.save:
    svpth = os.path.join(savedir, '{chosen}_latents.png')
    plot_latent(latent[0,:,:], target, svpth)  
else:
    plot_latent(latent[0,:,:], target)  


if args.fast:
    exit()

### compute gradient and plot or save 


if args.database == 'Toy':
    avg = np.zeros((128, 128) + (latent.shape[-1],), dtype = float)
    imout = np.zeros((128,128) + (3,), dtype = float)
    mask = avg[:,:,0] == 0 
else: 
    avg = np.zeros(data.LC.shape + (latent.shape[-1],), dtype = float)
    imout = np.zeros(data.LC.shape + (3,), dtype = float)
    mask = data.LC > 0  

print(x.shape)
tpb = model.tpb 
imout[:,:, 0] = x.detach().numpy()[0, -1, :, :, 0]
imout[:,:, 1] = x_out.detach().numpy()[0, -1, :, :, 0]
imout[:,:, 2] = (imout[:,:, 0] - imout[:,:, 1])**2
if args.grad:
    for j in range(latent.shape[-1]):
        grad = np.zeros(x.shape[1:]) 
        for i in range(tpb): 
            mu[i,j].backward(retain_graph = True)
            grad[i,:, :] += x.grad.numpy()[0,i,:,:]
            x.grad.fill_(0.0)
        avg[:,:,j] = grad.mean(0)[:,:,0]
        

if args.save:
    img = Image.fromarray(imout[:,:,1])
    svpth = os.path.join(savedir, f'{chosen}_reconstruction_lag={model.lag}.tiff')
    img.save(svpth) 
    for j in range(latent.shape[-1]):
        img = Image.fromarray(avg[:,:,j])
        svpth = os.path.join(savedir, f'{chosen}_grad_avg_latent{j}_lag={model.lag}.tiff')
        img.save(svpth)

else:
    plot_output(imout)
    plot_output(avg)

if args.extract:
    ## save latent
    np.savetxt(os.path.join(savedir, f'{chosen}_latents.csv'), latent.detach().numpy()[0,:,:])
    ## save target
    np.savetxt(os.path.join(savedir, f'{chosen}_target.csv'), target.detach().numpy()[0,:]) 


if args.nig:

    baseline = np.zeros(x.shape, dtype="float32")
    
    
    if args.database == 'AfricaNDVI':
       days = np.linspace(5, 365, 46, dtype = int)  
       template = 'databases/data/africa_ndvi_seasonality/ndvi_seasonality_TMSTMP.nc'
       paths = [os.path.join(template.replace('TMSTMP', f'{day:03}')) 
            for day in days]  
       for j, path in enumerate(paths):
           data = nc.Dataset(path)
           val = np.array(data['ndvi'])
           baseline[0, j + 46 * np.arange(3), :] = val[mask] 

    baseline = torch.Tensor(baseline)
    nig = NeuronIntegratedGradients(model, model.mu_layer, multiply_by_inputs = True)

    for j in range(1):
        # Baseline for Integrated Gradients
        # Zeros (default)
            
        # 1) NeuronIntegratedGradients, to see, for each latent representation,
        # the attribution to each of the input spatial locations in features (e.g. NDVI)

        attr_maps = nig.attribute(x,(j,), baselines=baseline, internal_batch_size=1)
        attr_maps = attr_maps
        os.makedirs(os.path.join(savedir, f'nig{j}'), exist_ok = True)
        for i in range(model.tpb):
            imgarray = np.zeros(mask.shape)
            imgarray[mask] = attr_maps.detach().numpy()[0,i,:]
            img = Image.fromarray(imgarray)
            img.save(os.path.join(savedir, f'nig{j}', f'{i}_.tiff'))
