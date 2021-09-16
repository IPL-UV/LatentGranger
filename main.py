#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LatentGranger Main


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

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Model
from model import LatentGranger

def main(config, arch, database):

    # Experiment ID
    experiment_id = str(datetime.now())
    if not os.path.isdir(config['trainer']['save_dir']):
        os.mkdir(config['trainer']['save_dir'])
    if not os.path.isdir(config['trainer']['save_dir']+'/'+database):
        os.mkdir(config['trainer']['save_dir']+'/'+database)
    if not os.path.isdir(config['trainer']['save_dir']+'/'+database+'/'+arch):
        os.mkdir(config['trainer']['save_dir']+'/'+database+'/'+arch)
    config['trainer']['save_dir'] = config['trainer']['save_dir']+'/'+database+'/'+arch+'/'+experiment_id
    if not os.path.isdir(config['trainer']['save_dir']):
        os.mkdir(config['trainer']['save_dir'])
    
    # Build model
    model = LatentGranger(config, database)
    copyfile('./model/model.py', config['trainer']['save_dir']+'/model.py')

    with open(config['trainer']['save_dir']+'/model.txt', 'w') as f:
        print(model, file=f)
        
    # Load initial weights
    if os.path.isfile(config['arch']['initial_weights']):
        checkpoint = torch.load(config['arch']['initial_weights'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('Initial weights loaded!')
        
    # Loggers
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    tb_logger = pl_loggers.TensorBoardLogger('logs/'+database, name='LatentGranger', version=experiment_id)
    # wandb_logger = pl_loggers.WandbLogger(save_dir='logs/', project='template')
    
    # Callbacks
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/'+database+'/'+arch+'/'+experiment_id,
                                          filename='{epoch}-{val_loss:.5f}',
                                          mode='min', monitor='val_loss',
                                          save_last=True, save_top_k=5)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=500, 
                                   verbose=True, mode='min', strict=True)
    callbacks = [checkpoint_callback]  #, early_stopping]

    # Trainer
    # Resume from checkpoint
    if not os.path.isfile(config['arch']['resume']):
        resume = None
    else:
        resume = config['arch']['resume']
        print('Resuming from checkpoint...')
    trainer = pl.Trainer(accumulate_grad_batches=1, callbacks=callbacks, 
                         gpus=0, auto_select_gpus=False,
                         #gpus=1, auto_select_gpus=True,
                         log_every_n_steps=10, logger=[tb_logger], 
                         max_epochs=config['trainer']['epochs'],
                         num_sanity_val_steps=2,
                         reload_dataloaders_every_epoch=True,
                         replace_sampler_ddp=False, resume_from_checkpoint=resume, 
                         val_check_interval=1.0, weights_summary='full',
                         limit_train_batches = 1) 
                
    # Training
    trainer.fit(model)
    
    # Test
    trainer.test(ckpt_path='best')

if __name__ == '__main__':
   
    # ArgParse
    parser = argparse.ArgumentParser(description="ArgParse")
    parser.add_argument('-c', '--config', default='configs/config.yaml', type=str,
                      help='config file path (default: configs/config.yaml)')
    parser.add_argument('-a', '--arch', default='LatentGranger', type=str,
                      help='arch name (default: LatentGranger)')
    parser.add_argument('-d', '--database', default='Toy', type=str,
                      help='database name (default: Toy)')
    parser.add_argument('-l', '--lag', default=-1, type=int,
                      help='lag (default: -1)')
    parser.add_argument('-b', '--beta', default=-1, type=float,
                      help='beta (default: -1)')

    args = parser.parse_args()

    # Load YAML config file into a dict variable
    with open(args.config) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    if args.beta >= 0:
        config['arch'][args.arch]['beta'] = args.beta
    if args.lag >= 0:
        config['arch'][args.arch]['lag'] = args.lag
    main(config, args.arch, args.database)
