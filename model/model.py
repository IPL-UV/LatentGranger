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
from utils import * 
from .loss import *

# Databases
from torch.utils.data import DataLoader
import databases

class LatentGranger(pl.LightningModule):
    def __init__(self, config, database):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Config
        self.config = config

        # Device
        self.user_device = self.config['data_loader']['device']

        # coefficient for granger regularization
        self.beta = float(self.config['arch']['LatentGranger']['beta'])
        # lag 
        self.lag = int(self.config['arch']['LatentGranger']['lag'])

        # read from config
        self.encoder_out = eval(self.config['arch']['LatentGranger']['encoder']['out_features'])
        self.decoder_out = eval(self.config['arch']['LatentGranger']['decoder']['out_features'])
        self.pin_memory = self.config['data_loader']['pin_memory']
        self.num_workers = int(self.config['data_loader']['num_workers'])
        self.stage = self.config['arch']['stage']

        # Define datasets
        data = getattr(databases, database) 
        self.data_train = data(self.config['data'][database], 'train', 'dense', self.stage) 
        self.data_val = data(self.config['data'][database], 'val', 'dense', self.stage) 
        self.data_test = data(self.config['data'][database], 'test', 'dense', self.stage) 

        self.batch_size = self.config['data'][database]['batch_size']
        self.tpb = self.config['data'][database]['tpb'] 
         

        ### define loss function
        self.loss_fun = nn.L1Loss()

        # Define model layers
        # Encoder
        self.encoder_layers = nn.ModuleList()
        ### mode should be always dense here this is the dense autoencoder
        mode = 'dense'
        if mode == 'dense':
            self.input_size =  self.config['data'][database]['dense_input_size']
        else:
            self.input_size = np.prod(eval(self.config['data'][database]['input_size']))

        in_ = self.input_size 
        for out_ in self.encoder_out:
            self.encoder_layers.append(nn.Linear(in_, out_))
            in_ = out_
        

        self.latent_layer = nn.BatchNorm1d(out_) 
        ## initialize weight matrix as orthogonal 
        torch.nn.init.orthogonal_(self.encoder_layers[-1].weight.data)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for out_ in self.decoder_out:
            self.decoder_layers.append(nn.Linear(in_, out_))                                     
            in_ = out_
                                       
        # Output
        self.output_layer = nn.Linear(in_, self.input_size)

        self.drop_layer = nn.Dropout(p=0.4)

        
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers,
                              pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers,
                              pin_memory=self.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers,
                              pin_memory=self.pin_memory)
    
    def forward(self, x):
        # Define forward pass
    
        # Reshape to (b_s*tpb, ...)
        x_shape_or = np.shape(x)[2:]
        x = torch.reshape(x, (self.batch_size*self.tpb,) + x_shape_or)
        
        # Encoder
        for i in np.arange(len(self.encoder_out)):
            #if i < 2:
            x = self.drop_layer(x)
            x = self.encoder_layers[i](x)
            x = nn.LeakyReLU(0.1)(x)
            

        # apply normalization
        #x = self.latent_layer(x)

        # Latent representation
        # Reshape to (b_s, tpb, latent_dim)
        x_latent = torch.reshape(x, (self.batch_size, self.tpb,-1))
        # Decoder
        for i in np.arange(len(self.decoder_out)):
            #if i > 0: ## no dropout on the latent input to the decoder 
            #x = self.drop_layer(x)
            x = self.decoder_layers[i](x)
            x = nn.LeakyReLU(0.1)(x)
                    
        # Output
        x = self.output_layer(x)
        
        # Reshape to (b_s, tpb, ...)
        x = torch.reshape(x, (self.batch_size, self.tpb,) + x_shape_or)
        
        return x, x_latent
    
    def training_step(self, batch, idx):
        x, target = batch

        # Define training step
        x_out, x_latent = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0]).to(self.user_device)
        
        base_loss = self.loss_fun(x,x_out)
        loss += base_loss
        
        # Granger loss
        if self.config['arch']['stage'] == 'causality':
            Granger_loss = self.beta * granger_loss(x_latent, target, maxlag = self.lag)
            loss += Granger_loss
        else:
            Granger_loss = torch.tensor([0.0])


        Orth_loss = orth_loss(self.encoder_layers[-1].weight) 
        #Orth_loss = uncor_loss(x_latent) 
        loss += Orth_loss 

        return {'loss': loss,
                'base_loss': base_loss.detach().item(),
                'Granger_loss': Granger_loss.detach().item(),
                'Orth_loss': Orth_loss.detach().item()}
    
    def training_epoch_end(self, training_step_outputs):
        # Loggers
        loss = [batch['loss'].detach().cpu().numpy() for batch in training_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"train": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('train_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        base_loss = np.array([batch['base_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("base_losses", {"train": np.nanmean(base_loss)}, global_step=self.current_epoch)
        Granger_loss = np.array([batch['Granger_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("Granger_losses", {"train": np.nanmean(Granger_loss)}, global_step=self.current_epoch)
        Orth_loss = np.array([batch['Orth_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("Orth_losses", {"train": np.nanmean(Orth_loss)}, global_step=self.current_epoch)
        

        
    def validation_step(self, batch, idx): 
        x, target = batch
        
        # Define training step
        x_out, x_latent = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0]).to(self.user_device)

        base_loss = self.loss_fun(x,x_out)
        loss += base_loss
        
        # Granger loss
        if self.config['arch']['stage'] == 'causality':
            Granger_loss = self.beta * granger_loss(x_latent, target, maxlag = self.lag)
            loss += Granger_loss
        else:
            Granger_loss = torch.tensor([0.0])


        Orth_loss = orth_loss(self.encoder_layers[-1].weight) 
        #Orth_loss = uncor_loss(x_latent) 
        loss += Orth_loss 

        return {'val_loss': loss,
                'val_base_loss': base_loss.detach().item(),
                'val_Granger_loss': Granger_loss.detach().item(),
                'val_Orth_loss': Orth_loss.detach().item()}

    def validation_epoch_end(self, validation_step_outputs):
        # Loggers
        loss = [batch['val_loss'].detach().cpu().numpy() for batch in validation_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"val": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('val_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        base_loss = np.array([batch['val_base_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("base_losses", {"val": np.nanmean(base_loss)}, global_step=self.current_epoch)
        Granger_loss = np.array([batch['val_Granger_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("Granger_losses", {"val": np.nanmean(Granger_loss)}, global_step=self.current_epoch)
        Orth_loss = np.array([batch['val_Orth_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("Orth_losses", {"val": np.nanmean(Orth_loss)}, global_step=self.current_epoch)
        

    def test_step(self, batch, idx):
        x, target = batch
        
        # Define training step
        x_out, x_latent = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0]).to(self.user_device)
        
        base_loss = self.loss_fun(x,x_out)
        loss += base_loss
        
        # Granger loss
        if self.config['arch']['stage'] == 'causality':
            Granger_loss = self.beta * granger_loss(x_latent, target, maxlag = self.lag)
            loss += Granger_loss
        else:
            Granger_loss = torch.tensor([0.0])
        
        Orth_loss = orth_loss(self.encoder_layers[-1].weight) 
        #Orth_loss = uncor_loss(x_latent) 
        loss += Orth_loss 

        # Loggers
        return {'test_loss': loss,
                'test_base_loss': base_loss.detach().item(),
                'test_Granger_loss': Granger_loss.detach().item(),
                'test_Orth_loss': Orth_loss.detach().item()}
    
    def test_epoch_end(self, test_step_outputs):
        # Loggers
        loss = [batch['test_loss'].detach().cpu().numpy() for batch in test_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"test": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('test_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        base_loss = np.array([batch['test_base_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("base_losses", {"test": np.nanmean(base_loss)}, global_step=self.current_epoch)
        Granger_loss = np.array([batch['test_Granger_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("Granger_losses", {"test": np.nanmean(Granger_loss)}, global_step=self.current_epoch)
        Orth_loss = np.array([batch['test_Orth_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("Orth_losses", {"test": np.nanmean(Orth_loss)}, global_step=self.current_epoch)
        

    def configure_optimizers(self):
        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config['optimizer']['lr'], weight_decay=self.config['optimizer']['weight_decay'])
        
        return optimizer
