#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDVI database with coupled ENSO

"""

import os
import numpy as np
from skimage.io import imread
from scipy.ndimage.filters import uniform_filter1d
import torch
import torch.utils.data
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

        self.input_size = tuple(config['input_size'])
        self.flat_input_size = config['flat_input_size']
        # Load Land Cover map
        self.mask = imread(config['mask']) > 0 

        # Load time window
        self.tpb = config['tpb']

        # self.years = np.linspace(2007, 2017, 1, dtype = int)
        self.years = config[mode]
        #self.days = np.linspace(5, 365, 46, dtype=int)
        self.days = np.linspace(1, 353, 23, dtype=int)

        # load template
        self.template = config['template']

        # Load feature(s) maps paths
        self.root = config['root']
        self.timestamps = np.array([f'{year}{day:03}' for year in self.years
                                   for day in self.days])

        self.paths = [os.path.join(self.root,
                      self.template.replace('TMSTMP', tmstmp))
                      for tmstmp in self.timestamps]

        # Load ENSO
        # --- COMPLETE ---
        ENSO = np.loadtxt(self.config['root_ENSO'], comments='#')
        # Mean filter to adapt ENSO resolution to feature resolution
        ENSO[:, 1] = uniform_filter1d(ENSO[:, 1], 16)
        # Select time period for experiments
        self.ENSO = []
        for year in self.years:
            whr = np.where(np.array([str(int(t))[:4]
                           for t in ENSO[:, 0]]) == str(year))[0]
            ENSO_tmp = ENSO[whr, :]
            ENSO_tmp = ENSO_tmp[self.days - 1, :]
            self.ENSO.append(ENSO_tmp)
        self.ENSO = np.concatenate(self.ENSO)[:, 1]
    
    def getAll(self):
        # Features
        x = ()
        # ENSO
        enso = torch.Tensor(self.ENSO)

        # NDVI
        img_paths = self.paths
        ndvi = np.zeros((len(img_paths),) + self.input_size + (1,), dtype="float32")
        for j, path in enumerate(img_paths):
            data = nc.Dataset(path)
            x = np.array(data['ndvi'])
            x[x == data['ndvi']._FillValue] = 0.0  # set NA to 0.0
            ndvi[j, :, :, 0] = x

        # flatten if processing_mode is flat
        if self.processing_mode == 'flat':
            ndvi = np.reshape(ndvi, (len(img_paths), -1))
            ndvi = ndvi[:, np.ndarray.flatten(self.mask)]

        ndvi = torch.Tensor(ndvi)
        return ndvi, enso
   
    def __getitem__(self, index):
        """Returns tuple (input, target) correspond to batch #idx."""
        # Features
        x = ()

        # ENSO
        enso = torch.Tensor(self.ENSO[index:index + self.tpb])

        # NDVI
        img_paths = self.paths[index:index + self.tpb]
        if self.processing_mode == 'flat':
            ndvi = np.zeros((self.tpb,) + (self.flat_input_size,), dtype="float32")
        else:
            ndvi = np.zeros((self.tpb,) + self.input_size + (1,), dtype="float32")

        for j, path in enumerate(img_paths):
            data = nc.Dataset(path)
            x = np.array(data['ndvi'])
            if self.processing_mode == 'flat':
                ndvi[j, :] = x[self.mask]
            else:
                ndvi[j, :, :, 0] = x

        ndvi = torch.Tensor(ndvi)
        return ndvi, enso

    def __len__(self):
        """
        Returns:
            (int): the length of the dataset.
        """
        return len(self.paths) - self.tpb + 1
