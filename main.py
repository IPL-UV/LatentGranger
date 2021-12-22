#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BSD 3-Clause License (see LICENSE file)
# Copyright (c) Image and Signaling Process Group (ISP) IPL-UV 2021
# All rights reserved.

"""
Main script to execute the Latent Granger autoencoder
"""

import os
import git
import argparse
import yaml
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Model
import archs
import loaders
import torch

#torch.autograd.set_detect_anomaly(True)
def main(args):

    # Load YAML config files  into a dict variable

    with open(f'configs/archs/{args.arch}.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python dictionary format
        arch_config = yaml.load(file, Loader=yaml.FullLoader)

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
    if repo.is_dirty():
        kg = input("WARNING: the current repo has not tracked changes\n" + 
              "if you continue, any saved results will" + 
              " not be correctly associated to a commit hash\n" + 
              "type yes(y) to continue anyway: ")
        if kg != "yes" and kg != "y": 
            return
    git_commit_sha = repo.head.object.hexsha[:7]

    experiment_id = str(datetime.now())
    log_dir = os.path.join(args.dir, 'logs', args.data, args.arch,
                           experiment_id)
    checkpoints_dir = os.path.join(args.dir, 'checkpoints',
                                   args.data, args.arch,
                                   experiment_id)

    os.makedirs(args.dir, exist_ok=True)
    pathlogfile = os.path.join(args.dir, 'log.txt')


    # Build model

    if arch_config['processing_mode'] == 'flat':
        input_size = data_config['flat_input_size']
    else:
        input_size = tuple(data_config['input_size'])

    model_class = getattr(archs, arch_config['class'])

    if args.seed >= 0:
        print(f'seed set to {args.seed}') 
        pl.utilities.seed.seed_everything(seed=args.seed)

    model = model_class(arch_config, input_size, data_config['tpb'],
                        args.maxlag, args.gamma)
    print(model)

    # Build data module
    datamodule_class = getattr(loaders, loader_config['class'])
    datamodule = datamodule_class(loader_config, data_config,
                                  arch_config['processing_mode'])

    # Loggers
    tb_logger = pl_loggers.TensorBoardLogger(log_dir, name=args.arch,
                                             version=experiment_id)

    # Callbacks
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir,
                                          filename='{epoch}-{val_loss:.5f}',
                                          mode='min', monitor='val_loss',
                                          save_last=True, save_top_k=5)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0, patience=10,
                                   verbose=False, mode='min', strict=True)

    callbacks = [checkpoint_callback]
    if args.earlystop:
        callbacks += [early_stopping]

    trainer = pl.Trainer.from_argparse_args(args, logger=[tb_logger],
                                            callbacks=callbacks)

    # Training
    trainer.fit(model, datamodule)

    # Test
    res = trainer.test(ckpt_path='best')
    mse_test = res[0]["mse_loss"]["test"].detach().numpy()
    granger_test = res[0]["granger_loss"]["test"].detach().numpy()

    with open(pathlogfile, "a") as logfile:
        logfile.write(f'{experiment_id},{git_commit_sha},{args.arch},' +
                      f'{args.data},{args.loader},{args.gamma},' +
                      f'{args.maxlag},{mse_test},{granger_test}\n')


if __name__ == '__main__':
    # ArgParse
    parser = argparse.ArgumentParser(description="ArgParse")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--arch', default='vae', type=str,
                        help='name of the architecture associated' +
                             'to a config file' +
                             'in configs/archs/')
    parser.add_argument('-d', '--data', default='toy', type=str,
                        help='database name (default: toy) associated to a ' +
                             'config file in configs/data/')
    parser.add_argument('--loader', default='base', type=str,
                        help='loaders name (default: base) associated ' +
                             'to a config file in configs/loaders/')
    parser.add_argument('--maxlag', default=1, type=int,
                        help='maxlag (default: 1)')
    parser.add_argument('-g', '--gamma', default=0, type=float,
                        help='gamma regulazier for granger' +
                              'penalty (default: 0)')
    parser.add_argument('--earlystop', action='store_true',
                        help='whether to use early stopping')
    parser.add_argument('--dir', default="experiment",
                        type=str, help='experiemnt directory')
    parser.add_argument('--seed', default=-1,
                        type=int, help='seed if >0')

    args = parser.parse_args()
    main(args)
