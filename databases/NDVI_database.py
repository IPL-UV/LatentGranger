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

    def __init__(self, config, mode='train', processing_mode='flat'):
        self.config = config
        self.mode = mode
        self.processing_mode = processing_mode

        
        self.input_size = config['input_size']
        # Load Land Cover map
        self.LC = imread(config['root_LC'])

        # Load time window
        self.tpb = config['tpb']

        #self.years = np.linspace(2007, 2017, 1, dtype = int) 
        self.years = config[mode]
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
            x[x == data['ndvi']._FillValue] = 0.0 ## set NA to the mean
            #x = (x - self.mean) / self.std 
            ndvi[j, :, :, 0] = x 
            
        
        if self.processing_mode == 'flat':
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
