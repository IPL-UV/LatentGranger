#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LatentGranger Model Class


Anonymized code submitted alongide 
the manuscript titled 
Learning Granger Causal Feature Representations 

please do not distribute
"""

import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from utils import * 
from .loss import *

from PIL import Image

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
        

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for out_ in self.decoder_out:
            self.decoder_layers.append(nn.Linear(in_, out_))                                     
            in_ = out_
                                       
        # Output
        self.output_layer = nn.Linear(in_, self.input_size)

        
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers,
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
            x = self.encoder_layers[i](x)
            #if i < len(self.encoder_out) - 1:
            x = nn.LeakyReLU(0.1)(x)
            
        # Latent representation
        # Reshape to (b_s, tpb, latent_dim)
        x_latent = torch.reshape(x, (self.batch_size, self.tpb,-1))
        # Decoder
        for i in np.arange(len(self.decoder_out)):
            x = self.decoder_layers[i](x)
            #if i < len(self.decoder_out) - 1: 
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
        
        MSE_loss = nn.MSELoss(reduction='mean')(x,x_out)
        loss += MSE_loss
        
        # Granger loss
        if self.config['arch']['stage'] == 'causality':
            Granger_loss = self.beta * granger_loss(x_latent, target, maxlag = self.lag)
            loss += Granger_loss
        else:
            Granger_loss = torch.tensor([0.0])
        
        return {'loss': loss,
                'MSE_loss': MSE_loss.detach().item(),
                'Granger_loss': Granger_loss.detach().item()}
    
    def training_epoch_end(self, training_step_outputs):
        # Loggers
        loss = [batch['loss'].detach().cpu().numpy() for batch in training_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"train": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('train_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        MSE_loss = np.array([batch['MSE_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("MSE_losses", {"train": np.nanmean(MSE_loss)}, global_step=self.current_epoch)
        Granger_loss = np.array([batch['Granger_loss'] for batch in training_step_outputs])
        self.logger.experiment[0].add_scalars("Granger_losses", {"train": np.nanmean(Granger_loss)}, global_step=self.current_epoch)
        
    def validation_step(self, batch, idx): 
        x, target = batch
        
        # Define training step
        x_out, x_latent = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0]).to(self.user_device)

        MSE_loss = nn.MSELoss(reduction='mean')(x,x_out)
        loss += MSE_loss
        
        # Granger loss
        if self.config['arch']['stage'] == 'causality':
            Granger_loss = self.beta * granger_loss(x_latent, target, maxlag = self.lag)
            loss += Granger_loss
        else:
            Granger_loss = torch.tensor([0.0])
        
        return {'val_loss': loss,
                'val_MSE_loss': MSE_loss.detach().item(),
                'val_Granger_loss': Granger_loss.detach().item()}

    def validation_epoch_end(self, validation_step_outputs):
        # Loggers
        loss = [batch['val_loss'].detach().cpu().numpy() for batch in validation_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"val": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('val_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        MSE_loss = np.array([batch['val_MSE_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("MSE_losses", {"val": np.nanmean(MSE_loss)}, global_step=self.current_epoch)
        Granger_loss = np.array([batch['val_Granger_loss'] for batch in validation_step_outputs])
        self.logger.experiment[0].add_scalars("Granger_losses", {"val": np.nanmean(Granger_loss)}, global_step=self.current_epoch)

    def test_step(self, batch, idx):
        x, target = batch
        
        # Define training step
        x_out, x_latent = self(x)
        
        # Compute loss
        loss = torch.tensor([0.0]).to(self.user_device)
        
        MSE_loss = nn.MSELoss(reduction='mean')(x,x_out)
        loss += MSE_loss
        
        # Granger loss
        if self.config['arch']['stage'] == 'causality':
            Granger_loss = self.beta * granger_loss(x_latent, target, maxlag = self.lag)
            loss += Granger_loss
        else:
            Granger_loss = torch.tensor([0.0])
        
        # Loggers
        return {'test_loss': loss,
                'test_MSE_loss': MSE_loss.detach().item(),
                'test_Granger_loss': Granger_loss.detach().item()}
    
    def test_epoch_end(self, test_step_outputs):
        # Loggers
        loss = [batch['test_loss'].detach().cpu().numpy() for batch in test_step_outputs]
        self.logger.experiment[0].add_scalars("losses", {"test": np.nanmean(loss)}, global_step=self.current_epoch)
        self.log('test_loss', np.nanmean(loss), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        MSE_loss = np.array([batch['test_MSE_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("MSE_losses", {"test": np.nanmean(MSE_loss)}, global_step=self.current_epoch)
        Granger_loss = np.array([batch['test_Granger_loss'] for batch in test_step_outputs])
        self.logger.experiment[0].add_scalars("Granger_losses", {"test": np.nanmean(Granger_loss)}, global_step=self.current_epoch)
        
    def configure_optimizers(self):
        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config['optimizer']['lr'], weight_decay=self.config['optimizer']['weight_decay'])
        
        return optimizer
