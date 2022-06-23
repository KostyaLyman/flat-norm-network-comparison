# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:27:09 2022

Author: Rounak Meyur

Description: Tutorial for using persistent diagram
"""

from itertools import product

import time
import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

from ripser import Rips
from persim import PersistenceImager


#%% Persistence Imager

# Printing a PersistenceImager() object will print its defining attributes
pimgr = PersistenceImager(pixel_size=0.5, birth_range=(0,2))
print(pimgr.resolution)


#%% Fit resolution
# The `fit()` method can be called on one or more (*,2) numpy arrays to 
# automatically determine the miniumum birth and persistence ranges needed to 
# capture all persistence pairs. The ranges and resolution are automatically 
# adjusted to accomodate the specified pixel size.
pimgr = PersistenceImager(pixel_size=0.5)
pdgms = [np.array([[0.5, 0.8], [0.7, 2.2], [2.5, 4.0]]),
         np.array([[0.1, 0.2], [3.1, 3.3], [1.6, 2.9]]),
         np.array([[0.2, 1.5], [0.4, 0.6], [0.2, 2.6]])]
pimgr.fit(pdgms, skew=True)
print(pimgr)
print(pimgr.resolution)