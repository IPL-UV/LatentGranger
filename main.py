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
import git
import numpy as np
import argparse, yaml
from datetime import datetime
from shutil import copyfile

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Model
import models 
import loaders


# ArgParse
parser = argparse.ArgumentParser(description="ArgParse")
parser.add_argument('-m', '--model', default='vae', type=str,
                  help='name of the model associated to a config file in configs/models/')
parser.add_argument('-d', '--data', default='toy', type=str,
                  help='database name (default: toy) associated to a config file in configs/data/')
parser.add_argument('--loader', default='base', type=str,
                  help='loaders name (default: base) associated to a config file in configs/loaders/')
parser.add_argument('--maxlag', default=1, type=int,
                  help='lag (default: 1)')
parser.add_argument('-g', '--gamma', default=0, type=float,
                  help='gamma regulazier for granger penalty (default: 0)')
parser.add_argument('--gpu', default = 0, type = int, help = 'number of GPUs (0 for only CPU)')

args = parser.parse_args()

# Load YAML config files  into a dict variable

with open(f'configs/models/{args.model}.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python dictionary format
    model_config = yaml.load(file, Loader=yaml.FullLoader)

with open(f'configs/loaders/{args.loader}.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python dictionary format
    loader_config = yaml.load(file, Loader=yaml.FullLoader)


with open(f'configs/data/{args.data}.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python dictionary format
    data_config = yaml.load(file, Loader=yaml.FullLoader)


# Experiment ID
repo = git.Repo(search_parent_directories=True)
git_commit_sha = repo.head.object.hexsha[:7]

experiment_id = str(datetime.now()) 
log_dir =  os.path.join('logs', args.data, args.model, git_commit_sha, experiment_id) 
checkpoints_dir =  os.path.join('logs', args.data, args.model, git_commit_sha, experiment_id) 

# Build model

model_class = getattr(models, model_config['class']) 
model = model_class(model_config, data_config['input'], data_config['tpb'],  args.maxlag, args.gamma)
print(model)

# Build data models
datamodule_class = getattr(loaders, loader_config['class']) 
datamodule = datamodule_class(loader_config, data_config)

    
# Loggers
# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
tb_logger = pl_loggers.TensorBoardLogger(log_dir , name = args.model, version = experiment_id)

# Callbacks
# Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(dirpath= checkpoints_dir,
                                      filename='{epoch}-{val_loss:.5f}',
                                      mode='min', monitor='val_loss',
                                      save_last=True, save_top_k=5)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10, 
                               verbose=True, mode='min', strict=True)
callbacks = [checkpoint_callback]  #, early_stopping]


trainer = pl.Trainer(accumulate_grad_batches=1, callbacks=callbacks, 
                     gpus=args.gpu, auto_select_gpus= args.gpu > 0,
                     log_every_n_steps=10, logger=[tb_logger], 
                     max_epochs=100,
                     num_sanity_val_steps=2,
                     weights_summary='full') 
            
# Training
trainer.fit(model, datamodule)

# Test
trainer.test(ckpt_path='best')
