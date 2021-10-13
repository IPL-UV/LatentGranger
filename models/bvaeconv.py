#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LatentGranger Model Class

BSD 3-Clause License (see LICENSE file)

Copyright (c) Image and Signaling Process Group (ISP) IPL-UV 2021, 
All rights reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .loss import *

# Databases
from torch.utils.data import DataLoader
import databases

class bvaeconv(pl.LightningModule):
    def __init__(self, config, input_size, tpb, maxlag = 1, gamma = 0.0):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Config
        self.config = config

        # coefficient for granger regularization
        self.gamma = float(gamma)

        # coefficient for beta-VAE
        self.beta = float(config['beta'])

        # lag 
        self.lag = int(maxlag)

        # read from config
        self.latent_dim = self.config['latent_dim']  
        self.encoder_out = tuple(self.config['encoder']['out_features'])
        self.decoder_out = tuple(self.config['decoder']['out_features'])

        self.tpb = tpb
        self.input_size = input_size

        ### define distribution for VAE
        self.N = torch.distributions.Normal(0,1) 

        # Define model layers
        # Encoder
        self.encoder_layers = nn.ModuleList()
        
        in_ = 1
        for out_ in self.encoder_out:
            self.encoder_layers.append(nn.Conv2d(in_, out_, kernel_size = 3, padding = 'same'))
            in_ = out_
        
        self.w_reduced = int( self.input_size[0] // (2**len(self.encoder_out)))
        self.h_reduced = int( self.input_size[1] // (2**len(self.encoder_out)))
        self.hw_reduced = self.w_reduced * self.h_reduced 

        self.flatten = nn.Flatten()
        self.mu_layer = nn.Linear(in_ * self.hw_reduced , self.latent_dim) 
        self.sigma_layer = nn.Linear(in_ * self.hw_reduced , self.latent_dim) 

        ## initialize weight matrix as orthogonal 
        torch.nn.init.orthogonal_(self.mu_layer.weight.data)

        in_ = self.latent_dim

        self.decoder_init = nn.Linear(in_, self.hw_reduced)

        # Decoder
        in_ = 1
        self.decoder_layers = nn.ModuleList()
        for out_ in self.decoder_out:
            self.decoder_layers.append(nn.Conv2d(in_, out_, kernel_size = 3, padding = 'same'))
            in_ = out_
                                       
        self.pool = nn.MaxPool2d(2 , 2)
        self.upsample = nn.Upsample(scale_factor = 2) 
        self.final_upsample = nn.Upsample(size = self.input_size) 

        
    def forward(self, x):
        # Define forward pass
    

        # Reshape to (b_s*tpb, ...)
        x_shape_or = np.shape(x)[2:]
        x = torch.reshape(x, (self.tpb,) + (1,) +  self.input_size)  
        
         
        # Encoder
        for i in np.arange(len(self.encoder_out)):
            x = self.encoder_layers[i](x)
            x = nn.LeakyReLU(0.01)(x)
            x = self.pool(x)

            
        x = self.flatten(x)

        mu = self.mu_layer(x)   
        sigma = torch.exp(self.sigma_layer(x)) 

        x = mu + sigma * self.N.sample(mu.shape) 
        
        # Latent representation
        # Reshape to (b_s, tpb, latent_dim)
        x_latent = torch.reshape(x, (1, self.tpb,-1))
       
        x = self.decoder_init(x) 
        x = torch.reshape(x, (self.tpb,) + (1,) + (self.w_reduced, self.h_reduced)) 

        # Decoder
        for i in np.arange(len(self.decoder_out)):
            x = self.decoder_layers[i](x)
            if i < len(self.decoder_out) - 1:
               x = nn.LeakyReLU(0.01)(x)
            x = self.upsample(x)
                   
        # final upsample to match initial size
        # this is needed if the original size is not divisible for 2**(num. layer. encoder/decoder)   
        x = self.final_upsample(x)

        # Reshape to (b_s, tpb, ...)
        x = torch.reshape(x, (1, self.tpb,) + x_shape_or)
        
        return x, x_latent, mu, sigma
    
    def training_step(self, batch, idx):
        x, target = batch

        # Define training step
        x_out, x_latent, mu, sigma = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0])
        

        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = nn.functional.mse_loss(x_out,x)
        loss += mse_loss + self.beta * kl_loss
        
        # Granger loss
        Granger_loss = granger_loss(x_latent, target, maxlag = self.lag)
        loss += self.gamma * Granger_loss


        return {'loss': loss,
                'mse_loss': mse_loss.detach().item(),
                'kl_loss' : kl_loss.detach().item(),
                'granger_loss': Granger_loss.detach().item()}
    
    def training_epoch_end(self, training_step_outputs):
        # Loggers
        loss = [batch['loss'].detach().cpu().numpy() for batch in training_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"train": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('train_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)

        mse_loss = np.array([batch['mse_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("mse_losses", {"train": np.nanmean(mse_loss)}, global_step=self.current_epoch)

        kl_loss = np.array([batch['kl_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("kl_losses", {"train": np.nanmean(kl_loss)}, global_step=self.current_epoch)

        Granger_loss = np.array([batch['granger_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("granger_losses", {"train": np.nanmean(Granger_loss)}, global_step=self.current_epoch)

        
    def validation_step(self, batch, idx): 
        x, target = batch
        
        # Define training step
        x_out, x_latent, mu, sigma = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0])

        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = nn.functional.mse_loss(x_out,x)
        loss += mse_loss + self.beta * kl_loss
        
        # Granger loss
        Granger_loss =  granger_loss(x_latent, target, maxlag = self.lag)
        loss += self.gamma * Granger_loss

        return {'val_loss': loss,
                'val_mse_loss': mse_loss.detach().item(),
                'val_kl_loss': kl_loss.detach().item(),
                'val_granger_loss': Granger_loss.detach().item()}

    def validation_epoch_end(self, validation_step_outputs):
        # Loggers
        loss = [batch['val_loss'].detach().cpu().numpy() for batch in validation_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"val": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('val_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)

        mse_loss = np.array([batch['val_mse_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("mse_losses", {"val": np.nanmean(mse_loss)}, global_step=self.current_epoch)

        kl_loss = np.array([batch['val_kl_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("kl_losses", {"val": np.nanmean(kl_loss)}, global_step=self.current_epoch)


        Granger_loss = np.array([batch['val_granger_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("granger_losses", {"val": np.nanmean(Granger_loss)}, global_step=self.current_epoch)


    def test_step(self, batch, idx):
        x, target = batch
        
        # Define training step
        x_out, x_latent, mu, sigma = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0])
        

        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = nn.functional.mse_loss(x_out,x)
        loss += mse_loss + self.beta * kl_loss       

        Granger_loss = granger_loss(x_latent, target, maxlag = self.lag)
        loss += self.gamma * Granger_loss
        

        # Loggers
        return {'test_loss': loss,
                'test_mse_loss': mse_loss.detach().item(),
                'test_kl_loss': kl_loss.detach().item(),
                'test_granger_loss': Granger_loss.detach().item()}
    
    def test_epoch_end(self, test_step_outputs):
        # Loggers
        loss = [batch['test_loss'].detach().cpu().numpy() for batch in test_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"test": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('test_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)

        mse_loss = np.array([batch['test_mse_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("mse_losses", {"test": np.nanmean(mse_loss)}, global_step=self.current_epoch)

        kl_loss = np.array([batch['test_kl_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("kl_losses", {"test": np.nanmean(kl_loss)}, global_step=self.current_epoch)


        Granger_loss = np.array([batch['test_granger_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("granger_losses", {"test": np.nanmean(Granger_loss)}, global_step=self.current_epoch)


    def configure_optimizers(self):
        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config['optimizer']['lr'], weight_decay=self.config['optimizer']['weight_decay'])
        
        return optimizer
