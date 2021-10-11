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

class bvae(pl.LightningModule):
    def __init__(self, config, input_size, tpb, maxlags = 1, gamma = 0.0):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Config
        self.config = config
        self.tpb = tpb 
         
        # coefficient for granger regularization
        self.gamma = float(gamma)

        # coefficient for beta-VAE
        self.beta = float(config['beta'])

        # lag 
        self.lag = int(maxlags)

        # read from config
        self.latent_dim = config['latent_dim']  
        self.encoder_out = eval(config['encoder']['out_features'])
        self.decoder_out = eval(config['decoder']['out_features'])

        ### define distribution for VAE
        self.N = torch.distributions.Normal(0,1) 

        self.input_size = input_size

        # Define model layers
        # Encoder
        self.encoder_layers = nn.ModuleList()

        in_ = self.input_size 
        for out_ in self.encoder_out:
            self.encoder_layers.append(nn.Linear(in_, out_))
            in_ = out_
        
        self.mu_layer = nn.Linear(in_, self.latent_dim) 
        self.sigma_layer = nn.Linear(in_, self.latent_dim) 

        in_ = self.latent_dim
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for out_ in self.decoder_out:
            self.decoder_layers.append(nn.Linear(in_, out_))                                     
            in_ = out_
                                       
        # Output
        self.output_layer = nn.Linear(in_, self.input_size)

        self.drop_layer = nn.Dropout(p=0.4)

        
    def forward(self, x):
        # Define forward pass
    
        # Reshape to (b_s*tpb, ...)
        x_shape_or = np.shape(x)[2:]
        x = torch.reshape(x, (self.tpb,) + x_shape_or)
        
        # Encoder
        for i in np.arange(len(self.encoder_out)):
            x = self.encoder_layers[i](x)
            x = F.leaky_relu(x, 0.01)
            
        mu = self.mu_layer(x)   
        sigma = torch.exp(self.sigma_layer(x)) 


        x = mu + sigma * self.N.sample(mu.shape) 
        
        # Latent representation
        # Reshape to (b_s, tpb, latent_dim)
        x_latent = torch.reshape(x, (1, self.tpb,-1))

        # Decoder
        for i in np.arange(len(self.decoder_out)):
            x = self.decoder_layers[i](x)
            x = F.leaky_relu(x, 0.01)
                    
        # Output
        x = self.output_layer(x)
        
        # Reshape to (b_s, tpb, ...)
        x = torch.reshape(x, (1, self.tpb,) + x_shape_or)
        
        return x, x_latent, mu, sigma
    
    def training_step(self, batch, idx):
        x, target = batch

        # Define training step
        x_out, x_latent, mu, sigma = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0])
        
        kl_loss = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = F.mse_loss(x_out,x)
        loss += mse_loss + kl_loss
        
        # Granger loss
        Granger_loss = self.gamma * granger_simple_loss(x_latent, target, maxlag = self.lag)
        loss += Granger_loss


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


        kl_loss = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = F.mse_loss(x_out,x)
        loss += mse_loss + kl_loss
        
        # Granger loss
        Granger_loss = self.gamma * granger_simple_loss(x_latent, target, maxlag = self.lag)
        loss += Granger_loss

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
        

        kl_loss = self.beta*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = F.mse_loss(x_out,x)
        loss += mse_loss + kl_loss       

        Granger_loss = self.gamma * granger_simple_loss(x_latent, target, maxlag = self.lag)
        loss += Granger_loss
        

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
