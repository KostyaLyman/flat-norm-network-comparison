# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Chooses particular regions in a geographic location to compare the
networks inside it.
"""

import sys, os
from matplotlib import pyplot as plt, axes
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import pandas as pd
import csv


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
# FNN = NORMALIZED_FLAT_NORM = "\\tilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"


workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from pyFlatNormFixture import FlatNormFixture
from pyDrawNetworklib import get_vertseg_geometry


# get fixture
fx = FlatNormFixture('runTest')
fx.out_dir = "out/test"
fx.fig_dir = "figs/test"



# read geometries
area = 'mcbryde'
act_geom, synt_geom, hull = fx.read_networks(area)
verts,_ = get_vertseg_geometry(act_geom)

# compute stats
flatnorm_data = {
    'epsilons': [], 'lambdas': [], 'flatnorms': [],
    'norm_lengths': [], 'norm_areas': [],
    'input_lengths': [], 'input_ratios': [],
}

# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    967: 'ex3',
    930: 'ex4'}

#%% Flat norm computation
def compute_flat_norm_region(ind,point,eps,lamb_):
    # Tag the point
    region = ind_label[ind]
    
    # compute flat norm
    norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
        fx.get_region(point, eps),
        act_geom, synt_geom,
        lambda_=lamb_,
        normalized=True,
        plot=True
    )
    
    titles = {
        'lambda': f"$\\lambda = {lambda_:d}$",
        'epsilon': f"$\\epsilon = {epsilon:0.4f}$",
        'ratio': f"$|T|/\\epsilon = {w/epsilon :0.3g}$",
        'fn': f"${FNN}={norm:0.3g}$",
    }
    title = ", ".join([titles[t_name] for t_name in titles])
    
    # plot flat norm
    fx.plot_triangulated_region_flatnorm(
        epsilon=eps, lambda_=lamb_,
        to_file=f"{area}-fn_region_{region}",
        suptitle=title,
        do_return=False, show=True,
        **plot_data
        )
    return

#%% compute flat norm

# parameters
epsilon = 1e-3
lambda_ = 1000

for ind in ind_label:
    compute_flat_norm_region(ind, verts[ind],epsilon,lambda_)




