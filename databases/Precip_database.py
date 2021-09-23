#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDVI database with coupled ENSO

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize, rotate
from scipy.ndimage.filters import uniform_filter1d
from scipy.io import loadmat
import torch
import torch.utils.data 
from natsort import natsorted
import netCDF4 as nc
eps = 1e-7

class precip(torch.utils.data.Dataset):
    """
    precip-ENSO database loader. 
    """

    def __init__(self, config, mode='train', processing_mode = 'dense'):
        self.config = config
        self.mode = mode
        self.processing_mode = processing_mode
        
        self.input_size = config['input_size']

        # Load Land Cover map
        self.LC = imread(config['root_LC'])

        # Load time window
        self.tpb = config['tpb']

        self.years = config['years']
        #days = np.linspace(5, 357, 45, dtype = int)  

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
        ENSO = np.loadtxt(self.config['root_ENSO'], comments='#')
        # Mean filter to adapt ENSO resolution to feature resolution
        #ENSO[:,1] = uniform_filter1d(ENSO[:,1], 8)

        
        self.target = [] 
        for year in self.years:
            ENSO_tmp = ENSO[np.where(np.array([str(int(t))[:4] for t in ENSO[:,0]])==str(year))[0],:]
            #ENSO_tmp = ENSO_tmp[days - 1,:]
            self.target.append(ENSO_tmp)
        self.target = np.concatenate(self.target)[:,1]

        # Select time period for experiments
        if self.mode == 'train':
            self.paths = self.paths[:self.nsamples_train]
            self.target = self.target[:self.nsamples_train]
        elif self.mode == 'val':
            self.target = self.target[self.nsamples_train:self.nsamples_train+self.nsamples_val]
            self.paths = \
                self.paths[self.nsamples_train:self.nsamples_train+self.nsamples_val]
        elif self.mode == 'test':
            self.target = self.target[self.nsamples_train+self.nsamples_val:self.nsamples_train+self.nsamples_val + self.nsamples_test]
            self.paths = \
                self.paths[self.nsamples_train+self.nsamples_val:self.nsamples_train+self.nsamples_val + self.nsamples_test]



    def __getitem__(self, index):
        """Returns tuple (input, target) correspond to batch #idx."""
        # Features
        x = ()
       
        # ENSO
        target = torch.Tensor(self.target[index * self.tpb : index * self.tpb + self.tpb])

        # precip 
        img_paths = self.paths[index * self.tpb : index * self.tpb + self.tpb]
        vals = np.zeros((self.tpb,) + eval(self.input_size) + (1,), dtype="float32")
        for j, path in enumerate(img_paths):
            data = nc.Dataset(path)
            x = np.array(data['precip'])
            x[x == data['precip']._FillValue] = 0 ## set NA to 0
            vals[j, :, :, 0] = x 
            
        
        if self.processing_mode == 'dense':
            # Discard background values according to LC and flatten if autoencoder is dense
            vals = np.reshape(vals, (self.tpb, -1))
            vals = vals[:,np.ndarray.flatten(self.LC)>0]

        return torch.Tensor(vals), target

    def __len__(self):
        """
        Returns:
            (int): the length of the dataset.
        """
        return len(self.paths) // self.tpb
