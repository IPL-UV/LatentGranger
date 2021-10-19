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
from losses import granger_loss


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

    def forward(self, x):
        # Define forward pass

        # Reshape to (b_s*tpb, ...)
        x_shape_or = np.shape(x)[2:]
        x = torch.reshape(x, (self.tpb,) + (1,) + self.input_size)

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
        x_latent = torch.reshape(x, (1, self.tpb, -1))

        x = self.decoder_init(x)
        out_shape = (self.tpb,) + (1,) + (self.w_reduced, self.h_reduced)
        x = torch.reshape(x, out_shape)

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
        mse_loss = F.mse_loss(x_out, x)
        loss += mse_loss + self.beta * kl_loss

        # Granger loss
        g_loss = 0
        for idx in range(self.causal_latents):
            g_loss += granger_loss(x_latent, target,
                                   maxlag=self.lag, idx=idx)
        loss += self.gamma * g_loss

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

        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = F.mse_loss(x_out, x)
        loss += mse_loss + self.beta * kl_loss

        # Granger loss
        g_loss = 0
        for idx in range(self.causal_latents):
            g_loss += granger_loss(x_latent, target,
                                   maxlag=self.lag, idx=idx)

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

        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        mse_loss = F.mse_loss(x_out, x)
        loss += mse_loss + self.beta * kl_loss

        g_loss = 0
        for idx in range(self.causal_latents):
            g_loss += granger_loss(x_latent, target,
                                   maxlag=self.lag, idx=idx)

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
        lr = self.config['optimizer']['lr']
        weight_decay = self.config['optimizer']['weight_decay']
        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad,
                                  self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr,
                                     weight_decay=weight_decay)

        return optimizer
