#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# BSD 3-Clause License (see LICENSE file)
# Copyright (c) Image and Signaling Process Group (ISP) IPL-UV 2021,
# All rights reserved.

"""
LatentGranger Model Class
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from losses import *


class bvae(pl.LightningModule):
    def __init__(self, config, input_size, tpb, maxlag=1, gamma=0.0):
        super().__init__()

        self.automatic_optimization = False
        # Save hyperparameters
        self.save_hyperparameters()

        # Config
        self.config = config
        self.tpb = tpb

        # coefficient for granger regularization
        self.gamma = float(gamma)
        self.causal_loss = granger_simple_loss

        # coefficient for beta-VAE
        self.beta = float(config['beta'])

        # lag
        self.lag = int(maxlag)

        # read from config
        self.latent_dim = config['latent_dim']
        self.causal_latents = config['causal_latents']
        if self.causal_latents > self.latent_dim:
            raise NameError('latent dimension should be greater or equal' +
                            ' then the number of causal latents, check' +
                            ' the arch config file.')
        self.encoder_out = config['encoder']['out_features']
        self.decoder_out = config['decoder']['out_features']

        # define distribution for VAE
        self.N = torch.distributions.Normal(0, 1)

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

        self.model0_layers = nn.ModuleList()
        self.model1_layers = nn.ModuleList()
        for idx in range(self.causal_latents):
            self.model0_layers.append(nn.Conv1d(1, 1, kernel_size = self.lag))
            self.model1_layers.append(nn.Conv1d(2, 1, kernel_size = self.lag))

        in_ = self.latent_dim
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for out_ in self.decoder_out:
            self.decoder_layers.append(nn.Linear(in_, out_))
            in_ = out_

        # Output
        self.output_layer = nn.Linear(in_, self.input_size)


    def encoder(self, x):

        # Encoder
        for i in np.arange(len(self.encoder_out)):
            x = self.encoder_layers[i](x)
            x = F.leaky_relu(x, 0.01)

        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        return mu, sigma

    def decoder(self, x): 

        # Decoder
        for i in np.arange(len(self.decoder_out)):
            x = self.decoder_layers[i](x)
            x = F.leaky_relu(x, 0.01)

        # Output
        x = self.output_layer(x)
        return x 
        

    def forward(self, x):

        # Reshape to (tpb, ...)
        x = torch.squeeze(x, dim = 0)

        # Define forward pass

        mu, sigma = self.encoder(x) 

        # sampling
        x = mu + sigma * self.N.sample(mu.shape)

        # Latent representation
        # Reshape to (b_s, tpb, latent_dim)
        x_latent = torch.reshape(x, (1, ) + x.shape)

 
        x = self.decoder(x) 
        # Reshape to (b_s, tpb, ...)
        x = torch.reshape(x, (1,) + x.shape)

        return x, x_latent, mu, sigma

    def training_step(self, batch, batch_idx):
        x, target = batch

        var_opt, main_opt = self.optimizers() 
        # Define training step
        x_out, x_latent, mu, sigma = self(x)

        # Compute loss
        loss = torch.tensor([0.0])

        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        reconstruction_loss = F.mse_loss(x_out, x, reduction='sum')
        mse_loss = reconstruction_loss / torch.numel(x)
        loss += reconstruction_loss + self.beta * kl_loss

        # Granger loss
        var_loss = torch.zeros(())
        for idx in range(self.causal_latents):
            xlat = mu[:, idx].reshape((1,1,-1)) 
            xtar = target[0,:].reshape((1,1,-1)) 
            pred0 = self.model0_layers[idx](xlat) 
            pred1 = self.model1_layers[idx](torch.cat((xlat,xtar), 1)) 
            loss0 = F.mse_loss(pred0[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            loss1 = F.mse_loss(pred1[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            var_loss += loss0 + loss1

        self.log('forecasting_loss', {"train": var_loss}, on_step=False,
                 on_epoch=True, logger=True)

        
        #gradient step for forecasting models 
        var_opt.zero_grad()
        self.manual_backward(var_loss, retain_graph=True)
        var_opt.step()

        g_loss = torch.zeros(())
        for idx in range(self.causal_latents):
            xlat = mu[:, idx].reshape((1,1,-1)) 
            xtar = target[0,:].reshape((1,1,-1)) 
            pred0 = self.model0_layers[idx](xlat) 
            pred1 = self.model1_layers[idx](torch.cat((xlat,xtar), 1)) 
            loss0 = F.mse_loss(pred0[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            loss1 = F.mse_loss(pred1[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            g_loss += torch.log(loss1 + loss0) - torch.log(loss0)


        loss += self.gamma * g_loss

        #main gradient step
        main_opt.zero_grad()
        self.manual_backward(loss, retain_graph=False)
        main_opt.step()
  
        self.log('loss', {"train": loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('mse_loss', {"train": mse_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('kl_loss', {"train": kl_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('granger_loss', {"train": g_loss}, on_step=False,
                 on_epoch=True, logger=True)
        return loss


    def validation_step(self, batch, idx):
        x, target = batch
        # Define training step
        x_out, x_latent, mu, sigma = self(x)
        # Compute loss
        loss = torch.tensor([0.0])

        kl_loss = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        reconstruction_loss = F.mse_loss(x_out, x, reduction='sum')
        mse_loss = reconstruction_loss / torch.numel(x)
        loss += reconstruction_loss + self.beta * kl_loss

        # Granger loss
        g_loss = torch.zeros(())
        var_loss = torch.zeros(())
        for idx in range(self.causal_latents):
            xlat = mu[:, idx].reshape((1,1,-1)) 
            xtar = target[0,:].reshape((1,1,-1)) 
            pred0 = self.model0_layers[idx](xtar) 
            pred1 = self.model1_layers[idx](torch.cat((xlat,xtar), 1)) 
            loss0 = F.mse_loss(pred0[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            loss1 = F.mse_loss(pred1[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            var_loss += loss0 + loss1
            g_loss += torch.log(loss1 + loss0) - torch.log(loss0)


        self.log('forecasting_loss', {"val": var_loss}, on_step=False,
                 on_epoch=True, logger=True)

        loss += self.gamma * g_loss

        self.log('loss', {"val": loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('mse_loss', {"val": mse_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('kl_loss', {"val": kl_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('granger_loss', {"val": g_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('val_loss', loss)

    def test_step(self, batch, idx):
        x, target = batch
        # Define training step
        x_out, x_latent, mu, sigma = self(x)
        # Compute loss
        loss = torch.tensor([0.0])

        kl_loss = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1/2).sum()
        reconstruction_loss = F.mse_loss(x_out, x, reduction='sum')
        mse_loss = reconstruction_loss / torch.numel(x)
        loss += reconstruction_loss + self.beta * kl_loss

        # Granger loss
        g_loss = torch.zeros(())
        for idx in range(self.causal_latents):
            xlat = mu[:, idx].reshape((1,1,-1)) 
            xtar = target[0,:].reshape((1,1,-1)) 
            pred0 = self.model0_layers[idx](xtar) 
            pred1 = self.model1_layers[idx](torch.cat((xlat,xtar), 1)) 
            loss0 = F.mse_loss(pred0[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            loss1 = F.mse_loss(pred1[:,:,:-1], xlat[:,:,self.lag:], reduction='mean')   
            g_loss += torch.log(loss1 + loss0) - torch.log(loss0)

        loss += self.gamma * g_loss

        self.log('loss', {"test": loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('mse_loss', {"test": mse_loss},
                 on_step=False, on_epoch=True, logger=True)
        self.log('kl_loss', {"test": kl_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('granger_loss', {"test": g_loss}, on_step=False,
                 on_epoch=True, logger=True)

    def configure_optimizers(self):
        # read parameters
        lr = self.config['optimizer']['lr']
        weight_decay = self.config['optimizer']['weight_decay']
        # build optimizers
        var_param = list(self.model0_layers.parameters()) +  list(self.model1_layers.parameters()) 
        var_opt = torch.optim.Adam(var_param, lr=lr, weight_decay=lr)
        main_param = list(self.encoder_layers.parameters()) + list(self.decoder_layers.parameters())
        main_param += list(self.mu_layer.parameters()) + list(self.sigma_layer.parameters()) 
        main_param += list(self.output_layer.parameters()) 
        main_opt = torch.optim.Adam(main_param, lr=lr, weight_decay=weight_decay)
        return var_opt, main_opt
