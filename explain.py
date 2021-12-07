#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BSD 3-Clause License (see LICENSE file)
# Copyright (c) Image and Signaling Process Group (ISP) IPL-UV 2021
# All rights reserved.

"""
Explain latent space of LatentGranger
"""

import os
import git
import numpy as np
import argparse, yaml
from datetime import datetime


import netCDF4 as nc
from natsort import natsorted
import torch
import pytorch_lightning as pl
from PIL import Image

import loaders
# Model
import archs 
from losses import lag_cor  
from losses import granger_loss, granger_simple_loss 
from utils import *

# PyTorch Captum for XAI
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance, LayerIntegratedGradients
from captum.attr import NeuronConductance, NeuronIntegratedGradients


# ArgParse
parser = argparse.ArgumentParser(description="ArgParse")

parser.add_argument('--arch', default='vae', type=str,
                  help='arch name (default: vae)')
parser.add_argument('-d', '--data', default='toy', type=str,
                  help='database name (default: toy)')
parser.add_argument('--loader', default='base', type=str,
                  help='loaders name (default: base) associated to a config file in configs/loaders/')
parser.add_argument('--dir', default='experiment', type=str,
                  help='path to experiment folder')
parser.add_argument('-c','--checkpoint', default='last.ckpt', type=str,
                  help='checkpoint (default: last)')
parser.add_argument('-t','--timestamp', default='', type=str,
                  help='timestampt (default:)')
parser.add_argument('--train', action = 'store_true',
                  help='use trainig data')
parser.add_argument('--val', action = 'store_true',
                  help='use val data')
parser.add_argument('--save', action = 'store_true',
                  help='save images')
parser.add_argument('--grad', action = 'store_true',
                  help='compute average gradient')
parser.add_argument('--extract', action = 'store_true',
                  help='extract latent series')
parser.add_argument('--nig', action = 'store_true',
                  help='run NIG')
parser.add_argument('--latint', action = 'store_true',
                  help='run LATENT INTERVENTIONS')
parser.add_argument('--idx', type=int, default = 0,
                  help='index of reconstruction to plot')


args = parser.parse_args()


log_root =  os.path.join(args.dir, 'logs', args.data, args.arch) 
check_root =  os.path.join(args.dir, 'checkpoints', args.data, args.arch) 

print(check_root)
allchckpts = natsorted(
            [
                fname
                for fname in os.listdir(check_root)
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

checkpoint = os.path.join(check_root, chosen, args.checkpoint)

print('chosen checkpoint: ' + checkpoint)

with open(f'configs/archs/{args.arch}.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python dictionary format
    arch_config = yaml.load(file, Loader=yaml.FullLoader)


## define the model and load the checkpoint
model = getattr(archs, arch_config['class']).load_from_checkpoint(checkpoint)

print("model loaded with chosen checkpoint") 

################### print model info #####################

print(model)
print(f'gamma: {model.gamma}, maxlag: {model.lag}')

#################### load data #########################

with open(f'configs/loaders/{args.loader}.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python dictionary format
    loader_config = yaml.load(file, Loader=yaml.FullLoader)


with open(f'configs/data/{args.data}.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python dictionary format
    data_config = yaml.load(file, Loader=yaml.FullLoader)

# Build data module
datamodule_class = getattr(loaders, loader_config['class']) 
datamodule = datamodule_class(loader_config, data_config, arch_config['processing_mode'])
 

#####################   here we can do inference, plot ##############


if args.train:
    data = datamodule.data_train
elif args.val:
    data = datamodule.data_val
else:
    data = datamodule.data_test

savedir = os.path.join('viz', chosen)
os.makedirs(savedir, exist_ok = True)

x, target = data.getAll() 
x = torch.reshape(x, (1,) + x.shape)
x.requires_grad_()
target = torch.reshape(target, (1,) + target.shape)

model.eval()
x_out, latent, mu, sigma = model(x)


for j in range(latent.shape[-1]):
    corr = lag_cor(latent[:,:,j], target, lag = model.lag) 
    print(f'lagged correlation with target of  latent {j}: {corr}')
    gloss = granger_loss(latent, target, maxlag = model.lag, idx=j)
    print(f'granger losss of latent {j}: {gloss}')


if args.save:
    svpth = os.path.join(savedir, f'{chosen}_latents.tiff')
    plot_latent(latent[0,:,:], target, svpth)  
else:
    plot_latent(latent[0,:,0:1], target)  


### prepare arrays 

if hasattr(data, 'mask'): 
   mask = data.mask 
   avg = np.zeros(data.mask.shape + (latent.shape[-1],), dtype = float)
   imout = np.zeros(data.mask.shape + (3,), dtype = float)
else:
   avg = np.zeros(data.input_size + (latent.shape[-1],), dtype = float) 
   imout = np.zeros(data.input_size + (3,), dtype = float)
   mask = avg[:,:,0] == 0 

tpb = model.tpb 

if arch_config['processing_mode']== 'flat':
    imout[mask, 0] = x.detach().numpy()[0, args.idx, :]
    imout[mask, 1] = x_out.detach().numpy()[0, args.idx, :]
    imout[mask, 2] = (imout[mask, 0] - imout[mask, 1])**2
else:
    imout[:, :, 0] = x.detach().numpy()[0, args.idx, :, :, 0]
    imout[:, :, 1] = x_out.detach().numpy()[0, args.idx, :, :, 0]
    imout[:, :, 2] = (imout[:, :, 0] - imout[:, :, 1])**2


if args.grad:
    for j in range(latent.shape[-1]):
        grad = np.zeros(x.shape[1:]) 
        for i in range(latent.shape[1]): 
            mu[i,j].backward(retain_graph = True)
            grad[i,:] += np.abs(x.grad.numpy()[0,i,:])
            #grad[i,:] += x.grad.numpy()[0,i,:]
            x.grad.fill_(0.0)
        avg[:,:,j][mask] = np.mean(grad, 0)
        #avg[:,:,j][mask] = np.amax(grad, 0)
        #avg[:,:,j] = grad.mean(0)[:,:,0]

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
    if args.grad: 
       plot_output(avg)


if args.extract:
    ## save latent
    np.savetxt(os.path.join(savedir, f'{chosen}_latents.csv'), latent.detach().numpy()[0,:,:])
    ## save target
    np.savetxt(os.path.join(savedir, f'{chosen}_target.csv'), target.detach().numpy()[0,:]) 


if args.nig:

    #baseline = np.zeros(x.shape, dtype="float32")
    
    #baseline = torch.Tensor(baseline)

    baseline = torch.mean(x[0,:,:], 0).expand(x.shape)
    nig = NeuronIntegratedGradients(model, model.mu_layer, multiply_by_inputs = True)

    for j in range(latent.shape[-1]):
        # Baseline for Integrated Gradients
        # Zeros (default)
            
        # 1) NeuronIntegratedGradients, to see, for each latent representation,
        # the attribution to each of the input spatial locations in features (e.g. NDVI)

        attr_maps = nig.attribute(x,(j,), baselines=baseline, internal_batch_size=1)
        attr_maps = attr_maps
        os.makedirs(os.path.join(savedir, f'nig{j}'), exist_ok = True)
        for i in range(attr_maps.shape[1]):
            imgarray = np.zeros(mask.shape)
            imgarray[mask] = attr_maps.detach().numpy()[0,i,:]
            img = Image.fromarray(imgarray)
            img.save(os.path.join(savedir, f'nig{j}', f'{i}_.tiff'))

if args.latint:
    avg.fill(0) ##clean avg array
    avg_max = avg.copy()
    avg_min = avg.copy()
    latent = torch.squeeze(latent, 0)
    latent_max = torch.amax(latent, 0)  
    latent_min = torch.amin(latent, 0)
    std, m = torch.std_mean(latent, 0)
    for j in range(latent.shape[1]):
        latent_int_max = latent.clone() 
        #latent_int_max[:,j] = latent_max[j]
        latent_int_max[:,j] += 0.5 * std[j]
        latent_int_min = latent.clone() 
        #latent_int_min[:,j] = latent_min[j]
        latent_int_min[:,j] -= 0.5 * std[j]
        out_int_max = model.decoder(latent_int_max)
        out_int_min = model.decoder(latent_int_min)
        diff_int_max = out_int_max - x_out[0,:,:] #/ (x_out[0,:,:]))
        diff_int_min = out_int_min - x_out[0,:,:] #/ (x_out[0,:,:]))
        avg_max[:,:,j][mask] = np.mean(diff_int_max.detach().numpy(), 0)
        avg_min[:,:,j][mask] = np.mean(diff_int_min.detach().numpy(), 0)
    if args.save:
        for j in range(latent.shape[-1]):
            img_max = Image.fromarray(avg_max[:,:,j])
            img_min = Image.fromarray(avg_min[:,:,j])
            svpth = os.path.join(savedir, f'{chosen}_latint_max_avg_latent{j}_lag={model.lag}.tiff')
            img_max.save(svpth)
            svpth = os.path.join(savedir, f'{chosen}_latint_min_avg_latent{j}_lag={model.lag}.tiff')
            img_min.save(svpth)
    else:
        plot_output(avg_max)
        plot_output(avg_min)
