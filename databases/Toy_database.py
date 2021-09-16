#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy Database

"""

import os
import numpy as np
import csv
from skimage.io import imread
import torch
import torch.utils.data 
from natsort import natsorted
eps = 1e-7

class Toy(torch.utils.data.Dataset):
    """
    Toy database loader. 
    """

    def __init__(self, config, mode='train', processing_mode = 'dense', stage = 'causality'):
        self.config = config
        self.mode = mode
        self.processing_mode = processing_mode
        self.root = self.config['root']
        self.targetpath = os.path.join(self.root, "target.txt")
        self.nsamples_train = self.config['nsamples_train']
        self.nsamples_test = self.config['nsamples_test']
        self.nsamples_val = self.config['nsamples_val']
        self.tbp = self.config['tpb']
        self.input_size = self.config['input_size']
        
        
        # Train/Validation/Test sets
        self._input_img_paths = natsorted(
            [
                os.path.join(self.root, fname)
                for fname in os.listdir(self.config['root'])
                if fname.endswith(".png")
            ],
        )

        target = np.loadtxt(self.targetpath)
        
        
        if self.mode == 'train':
            self._input_img_paths = self._input_img_paths[:self.nsamples_train]
            self.target = target[:self.nsamples_train]
        elif self.mode == 'val':
            self.target = target[self.nsamples_train:self.nsamples_train+self.nsamples_val]
            self._input_img_paths = \
                self._input_img_paths[self.nsamples_train:self.nsamples_train+self.nsamples_val]
        elif self.mode == 'test':
            self.target = target[self.nsamples_train+self.nsamples_val:self.nsamples_train+self.nsamples_val + self.nsamples_test]
            self._input_img_paths = \
                self._input_img_paths[self.nsamples_train+self.nsamples_val:self.nsamples_train+self.nsamples_val + self.nsamples_test]


    def __getitem__(self, index):
        """Returns tuple (input, target) correspond to batch #idx."""
        img_paths = self._input_img_paths[index : index + self.tbp]
        x = np.zeros((self.tbp,) + eval(self.input_size) + (1,), dtype="float32")
        y = self.target[index : index + self.tbp]
        for j, path in enumerate(img_paths):
            img = imread(path)
            x[j, :, :, 0] = img / 255.0
        if self.processing_mode == 'dense':
            x = np.reshape(x, (self.tbp, -1))
        x = torch.Tensor(x)
        y = torch.Tensor(y)


        return x, y

    def __len__(self):
        """
        Returns:
            (int): the length of the dataset.
        """
        return len(self._input_img_paths) - self.tbp + 1 
