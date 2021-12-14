#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMOS database with coupled ENSO

"""

import os
import numpy as np
import torch
import torch.utils.data 
from natsort import natsorted
import netCDF4 as nc
eps = 1e-7

class smos(torch.utils.data.Dataset):

    def __init__(self, config, mode='train', processing_mode = 'flat'):

        self.config = config
        self.mode = mode
        self.processing_mode = processing_mode
        
        self.input_size = tuple(config['input_size'])

        # Load time window
        self.tpb = config['tpb']

        self.nsamples_train = self.config['nsamples_train']
        self.nsamples_test = self.config['nsamples_test']
        self.nsamples_val = self.config['nsamples_val']



        # Load feature(s) maps paths
        self.root = config['root'] 

        self.paths = natsorted(
            [
                os.path.join(self.root, fname)
                for fname in os.listdir(self.config['root'])
                if fname.endswith(".nc")
            ],
        )

        # Load ENSO
        enso = nc.Dataset(self.config['target'])
        self.target = enso['ENSO4'] 

        # Select time period for experiments
        if self.mode == 'train':
            first = 0
            last = self.nsamples_train
        elif self.mode == 'val':
            first = self.nsamples_train
            last = self.nsamples_train + self.nsamples_val
        elif self.mode == 'test':
            first = self.nsamples_train+self.nsamples_val
            last = self.nsamples_train+self.nsamples_val + self.nsamples_test

        self.paths = self.paths[first:last]
        self.target = self.target[first:last]

        # load mask
        mask = nc.Dataset(self.config['mask'])
        self.mask = np.array(mask['mask']) > 0

    def getAll(self):
        # Features
        x = ()
        # ENSO
        target = torch.Tensor(self.target)

        # NDVI
        img_paths = self.paths
        vals = np.zeros((len(img_paths),) + self.input_size + (1,), dtype="float32")
        for j, path in enumerate(img_paths):
            data = nc.Dataset(path)
            x = np.array(data['smos'])
            x[x == data['smos']._FillValue] = 0.0  # set NA to 0.0
            vals[j, :, :, 0] = x

        # flatten if processing_mode is flat
        if self.processing_mode == 'flat':
            vals = np.reshape(ndvi, (len(img_paths), -1))
            vals = vals[:, np.ndarray.flatten(self.mask)]

        vals = torch.Tensor(vals)
        return vals, target
 

    def __getitem__(self, index):
        """Returns tuple (input, target) correspond to batch #idx."""
        # Features
        x = ()
       
        # ENSO
        target = torch.Tensor(self.target[index : index + self.tpb])

        # smos 
        img_paths = self.paths[index : index  + self.tpb]
        vals = np.zeros((self.tpb,) + self.input_size, dtype="float32")
        for j, path in enumerate(img_paths):
            data = nc.Dataset(path)
            x = np.array(data['smos'])
            x[x == data['smos']._FillValue] = 0 ## set NA to 0
            vals[j, :, :] = x 
            
        
        if self.processing_mode == 'flat':
            # flatten
            vals = np.reshape(vals, (self.tpb, -1))
            vals = vals[:, np.ndarray.flatten(self.mask)] 

        return torch.Tensor(vals), target

    def __len__(self):
        """
        Returns:
            (int): the length of the dataset.
        """
        return len(self.paths) - self.tpb + 1
