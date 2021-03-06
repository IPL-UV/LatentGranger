#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BSD 3-Clause License (see LICENSE file)

# Copyright (c) Image and Signaling Process Group (ISP) IPL-UV 2021,
# All rights reserved.

"""
LatentGranger Convolution Model Class
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class bvaeconv(pl.LightningModule):

    def __init__(self, config, input_size, tpb, maxlag=1, gamma=0.0):

        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Config
        self.config = config

        # coefficient for granger regularization
        self.gamma = float(gamma)

        # coefficient for beta-VAE
        self.beta = float(config['beta'])

        # index of the most causal latent
        self.causalix = int(0)

        # lag
        self.lag = int(maxlag)

        # read from config
        self.latent_dim = self.config['latent_dim']
        self.causal_latents = config['causal_latents']
        if self.causal_latents > self.latent_dim:
            raise NameError('latent dimension should be greater or equal' +
                            ' then the number of causal latents, check' +
                            ' the arch config file.')

        self.encoder_out = tuple(self.config['encoder']['out_features'])
        self.decoder_out = tuple(self.config['decoder']['out_features'])

        self.tpb = tpb
        self.input_size = input_size

        # define distribution for VAE
        self.N = torch.distributions.Normal(0, 1)

        # Define model layers
        # Encoder
        self.encoder_layers = nn.ModuleList()

        in_ = 1
        for out_ in self.encoder_out:
            self.encoder_layers.append(nn.Conv2d(in_, out_,
                                                 kernel_size=3,
                                                 padding='same'))
            in_ = out_

        self.w_reduced = int(self.input_size[0] // (2**len(self.encoder_out)))
        self.h_reduced = int(self.input_size[1] // (2**len(self.encoder_out)))
        self.hw_reduced = self.w_reduced * self.h_reduced

        self.flatten = nn.Flatten()
        self.mu_layer = nn.Linear(in_ * self.hw_reduced, self.latent_dim)
        self.sigma_layer = nn.Linear(in_ * self.hw_reduced, self.latent_dim)


        # forecasting layers
        self.model0_layers = nn.ModuleList()
        self.model1_layers = nn.ModuleList()
        for idx in range(self.causal_latents):
            self.model0_layers.append(nn.Conv1d(1, 1, kernel_size = self.lag))
            self.model1_layers.append(nn.Conv1d(2, 1, kernel_size = self.lag))

        in_ = self.latent_dim

        self.decoder_init = nn.Linear(in_, self.hw_reduced)
        # Decoder
        in_ = 1
        self.decoder_layers = nn.ModuleList()
        for out_ in self.decoder_out:
            self.decoder_layers.append(nn.Conv2d(in_, out_, kernel_size=3,
                                                 padding='same'))
            in_ = out_

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.final_upsample = nn.Upsample(size=self.input_size)
        # normalization constant
        self.NC = self.input_size * self.tpb 


    def encoder(self, x):
        # Encoder
        for i in np.arange(len(self.encoder_out)):
            x = self.encoder_layers[i](x)
            x = nn.LeakyReLU(0.01)(x)
            x = self.pool(x)

        x = self.flatten(x)

        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        return mu, sigma

    def decoder(self, x):
        x = self.decoder_init(x)
        # Decoder
        for i in np.arange(len(self.decoder_out)):
            x = self.decoder_layers[i](x)
            if i < len(self.decoder_out) - 1:
                x = nn.LeakyReLU(0.01)(x)
            x = self.upsample(x)

        # final upsample to match initial size
        # this is needed if the original size is not divisible
        # for 2**(num. layer. encoder/decoder)
        x = self.final_upsample(x)
        return x

    def forward(self, x):
        # Define forward pass

        # Reshape to (b_s*tpb, ...)
        x_shape_or = np.shape(x)[2:]
        x = torch.reshape(x, (self.tpb,) + (1,) + self.input_size)

        mu, sigma = self.encoder(x)
        x = mu + sigma * self.N.sample(mu.shape)

        # Latent representation
        # Reshape to (b_s, tpb, latent_dim)
        x_latent = torch.reshape(x, (1, self.tpb, -1))

        x = self.decoder(x)
        out_shape = (self.tpb,) + (1,) + (self.w_reduced, self.h_reduced)
        x = torch.reshape(x, out_shape)

        # Reshape to (b_s, tpb, ...)
        x = torch.reshape(x, (1, self.tpb,) + x_shape_or)

        return x, x_latent, mu, sigma

    def elbo(self, x, x_out, mu, sigma):
        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() / self.NC 
        reconstruction_loss = F.mse_loss(x_out, x, reduction='sum') / self.NC
        loss = reconstruction_loss + self.beta * kl_loss
        return loss, reconstruction_loss, kl_loss

    def granger_loss(self, mu, target):
        g_losses = torch.zeros((self.causal_latents,))
        var_loss = torch.zeros(())
        for idx in range(self.causal_latents):
            xlat = mu[:, idx].reshape((1,1,-1)) 
            xtar = target[0,:].reshape((1,1,-1)) 
            pred0 = self.model0_layers[idx](xlat) 
            pred1 = self.model1_layers[idx](torch.cat((xlat,xtar), 1)) 
            loss0 = F.mse_loss(pred0[:,:,:-1], xlat[:,:,self.lag:],
                               reduction='mean')   
            loss1 = F.mse_loss(pred1[:,:,:-1], xlat[:,:,self.lag:],
                               reduction='mean')   
            var_loss += loss0 + loss1
            #g_losses[idx] +=  loss1 / torch.abs(loss0 - loss1)
            g_losses[idx] +=  loss1 / loss0
            #g_losses[idx] += (loss1 - loss0) / (loss1 + loss0)
            #g_losses[idx] += torch.log(loss1 + loss0) - torch.log(loss0)

        return g_losses, var_loss


    def training_step(self, batch, batch_idx):
        x, target = batch

        var_opt, main_opt = self.optimizers() 
        # Define training step
        x_out, x_latent, mu, sigma = self(x)

        # Compute beta-elbo loss
        loss, mse_loss, kl_loss = self.elbo(x, x_out, mu, sigma)

        # forecasting loss
        g_loss, var_loss = self.granger_loss(mu, target)
        self.log('forecasting_loss', {"train": var_loss}, on_step=False,
                 on_epoch=True, logger=True)

        #gradient step for forecasting models 
        var_opt.zero_grad()
        self.manual_backward(var_loss, retain_graph=True)
        var_opt.step()

        #granger loss
        g_loss, var_loss = self.granger_loss(mu, target)
        self.causalix = int(torch.argmin(g_loss).numpy())
        #loss += self.gamma * g_loss[self.causalix]
        loss += self.gamma * torch.sum(g_loss)

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
        self.log('granger_loss', {"train": torch.sum(g_loss)}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('causalix', {"train": self.causalix}, on_step=True,
                 on_epoch=False, logger=True)
        return loss


    def validation_step(self, batch, idx):
        x, target = batch
        # Define training step
        x_out, x_latent, mu, sigma = self(x)

        # Compute loss
        loss, mse_loss, kl_loss = self.elbo(x, x_out, mu, sigma)

        # Granger loss
        g_loss, var_loss = self.granger_loss(mu, target)

        self.log('forecasting_loss', {"val": var_loss}, on_step=False,
                 on_epoch=True, logger=True)

        #loss += self.gamma * g_loss[self.causalix]
        loss += self.gamma *torch.sum(g_loss)

        self.log('loss', {"val": loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('mse_loss', {"val": mse_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('kl_loss', {"val": kl_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('granger_loss', {"val": torch.sum(g_loss)}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('val_loss', loss)
        self.log('val_granger_loss', torch.sum(g_loss))
        self.log('forecasting_loss_val', var_loss)
        #return var_loss

    #def validation_epoch_end(self, validation_step_outputs):
    #    var_loss = torch.stack(validation_step_outputs).mean()
    #    schd = self.lr_schedulers()
    #    schd.step(var_loss) 


    def test_step(self, batch, idx):
        x, target = batch
        # Define training step
        x_out, x_latent, mu, sigma = self(x)
        # Compute loss
        loss = torch.tensor([0.0])

        # Compute loss
        loss, mse_loss, kl_loss = self.elbo(x, x_out, mu, sigma)

        # Granger loss
        g_loss, var_loss = self.granger_loss(mu, target)

        # combine losses
        #loss += self.gamma * g_loss[self.causalix]
        loss += self.gamma * torch.sum(g_loss)

        self.log('loss', {"test": loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('mse_loss', {"test": mse_loss},
                 on_step=False, on_epoch=True, logger=True)
        self.log('kl_loss', {"test": kl_loss}, on_step=False,
                 on_epoch=True, logger=True)
        self.log('granger_loss', {"test": torch.sum(g_loss)}, on_step=False,
                 on_epoch=True, logger=True)

    def configure_optimizers(self):
        # read parameters
        lr = self.config['optimizer']['lr']
        weight_decay = self.config['optimizer']['weight_decay']
        # build optimizers
        var_param = list(self.model0_layers.parameters()) + \
                    list(self.model1_layers.parameters()) 
        var_opt = torch.optim.Adam(var_param, lr=lr,
                                   weight_decay=0.01)
        main_param = list(self.encoder_layers.parameters()) + \
                list(self.decoder_layers.parameters())
        main_param += list(self.mu_layer.parameters()) + \
                list(self.sigma_layer.parameters()) 
        main_param += list(self.decoder_init.parameters()) 
        main_opt = torch.optim.Adam(main_param, lr=lr, 
                                    weight_decay=weight_decay)
        return var_opt, main_opt
