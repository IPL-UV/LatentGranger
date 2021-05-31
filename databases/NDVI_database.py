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

class NDVI(torch.utils.data.Dataset):
    """
    NDVI-ENSO database loader. 
    """

    def __init__(self, config, mode='train', processing_mode = 'dense', arch_stage = 'causality'):
        self.config = config
        self.mode = mode
        self.processing_mode = processing_mode

        
        self.input_size = config['input_size']
        # Load Land Cover map
        self.LC = imread(config['root_LC'])

        # Load time window
        self.tpb = config['tpb']

        self.years = np.linspace(2007, 2017, 1, dtype = int) 
        self.years = config[arch_stage][mode]
        self.days = np.linspace(5, 365, 46, dtype = int)  

        # load template 
        self.template = config['template'] 

        # Load feature(s) maps paths
        self.root = config['root'] 
        self.timestamps = np.array([f'{year}{day:03}' for year in self.years for day in self.days])

        self.paths = [os.path.join(self.root, self.template.replace('TMSTMP', tmstmp)) 
            for tmstmp in self.timestamps]  

        # Load ENSO
        # --- COMPLETE ---
        ENSO = np.loadtxt(self.config['root_ENSO'], comments='#')
        # Mean filter to adapt ENSO resolution to feature resolution
        ENSO[:,1] = uniform_filter1d(ENSO[:,1], 8)
        # Select time period for experiments
        self.ENSO = [] 
        for year in self.years:
            ENSO_tmp = ENSO[np.where(np.array([str(int(t))[:4] for t in ENSO[:,0]])==str(year))[0],:]
            ENSO_tmp = ENSO_tmp[self.days - 1,:]
            self.ENSO.append(ENSO_tmp)
        self.ENSO = np.concatenate(self.ENSO)[:,1]

        # Compute statistics for normalization
        # Normalization
        if self.mode == 'train': 
            self._compute_statistics()
        else:
            statistics = np.load(self.config['statistics_root'])
            self.mean = statistics['mean']
            self.std = statistics['std']
        #self.mean = [0.0]
        #self.std = [1.0]

    def _compute_statistics(self):
        print("Computing statistics...")

        # Compute mean
        self.mean = 0.0
        self.std = 0.0
        nSamples_total = 0 
        for j, path in enumerate(self.paths):
            data = nc.Dataset(path)
            x = np.array(data['ndvi'])
            ix = ~(x == data['ndvi']._FillValue)
            nSamples_total += np.sum(ix)
            self.mean += np.sum(x[ix])
        self.mean = self.mean / nSamples_total
        
        # Compute standard deviation
        for j, path in enumerate(self.paths):
            data = nc.Dataset(path)
            x = np.array(data['ndvi'])
            ix = ~(x == data['ndvi']._FillValue)
            nSamples_total += np.sum(ix)
            self.std += np.sum(np.square(x[ix] - self.mean))
        self.std = self.std / (nSamples_total-1)
        self.std = np.sqrt(self.std)

        np.savez(self.config['statistics_root'], mean=self.mean, std=self.std)
       
    def __getitem__(self, index):
        """Returns tuple (input, target) correspond to batch #idx."""
        # Features
        x = ()
       
        # ENSO
        enso = torch.Tensor(self.ENSO[index : index + self.tpb])

        # NDVI 
        img_paths = self.paths[index : index + self.tpb]
        ndvi = np.zeros((self.tpb,) + eval(self.input_size) + (1,), dtype="float32")
        for j, path in enumerate(img_paths):
            data = nc.Dataset(path)
            x = np.array(data['ndvi'])
            x[x == data['ndvi']._FillValue] = self.mean ## set NA to the mean
            x = (x - self.mean) / self.std 
            ndvi[j, :, :, 0] = x 
            
        
        if self.processing_mode == 'dense':
            # Discard background values according to LC and flatten if autoencoder is dense
            ndvi = np.reshape(ndvi, (self.tpb, -1))
            ndvi = ndvi[:,np.ndarray.flatten(self.LC)>0]

        ndvi = torch.Tensor(ndvi) 
        return ndvi, enso

    def __len__(self):
        """
        Returns:
            (int): the length of the dataset.
        """
        return len(self.paths) - self.tpb + 1
