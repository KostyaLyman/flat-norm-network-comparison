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
from pprint import pprint


workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from pyFlatNormFixturelib import FlatNormFixture
from pyDrawNetworklib import get_vertseg_geometry


# get fixture
fx = FlatNormFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"



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

#%% Flat norm computation
def compute_flat_norm_region(ind,point,eps,lamb_):
    # Tag the point
    region = "chosen-"+str(ind)
    
    # compute flat norm
    norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
        fx.get_region(point, eps),
        act_geom, synt_geom,
        lambda_=lamb_,
        normalized=True,
        plot=True,
        compare_hausdorff=False
    )
    
    # plot flat norm
    fx.plot_triangulated_region_flatnorm(
        epsilon=eps, lambda_=lamb_,
        to_file=f"{area}-fn_region_{region}-eps_{epsilon:0.4f}",
        suptitle_sfx=f"$\\epsilon$={epsilon:0.5f} : |T| = {w:0.5f} : |T| / $\\epsilon$ = {w / epsilon:0.4f} \n $\lambda$ = {lambda_} : FN = {norm:0.5f} ",
        do_return=False, show=True,
        **plot_data
        )
    
    # Record statistics
    # flatnorm_data['epsilons'].append(f"{eps:0.4f}")
    # flatnorm_data['lambdas'].append(lamb_)
    # flatnorm_data['flatnorms'].append(norm)
    # flatnorm_data['norm_lengths'].append(enorm)
    # flatnorm_data['norm_areas'].append(tnorm)
    # flatnorm_data['input_lengths'].append(w)
    # flatnorm_data['input_ratios'].append(w/epsilon)
    return

#%% compute flat norm

# parameters
epsilon = 1e-3
lambda_ = 1000

ind = 930
compute_flat_norm_region(ind, verts[ind],epsilon,lambda_)

sys.exit(0)
#%%
for ind,pt in enumerate(verts):
    compute_flat_norm_region(ind, pt, epsilon, lambda_)
    
flatnorm_data = pd.DataFrame(flatnorm_data)

file_name = f"{area}-flatnorm-stats_{len(verts)}_regions"

with open(f"{fx.out_dir}/{file_name}.csv", "w") as outfile:
    flatnorm_data.to_csv(outfile, sep=",", index=False, header=True, 
                         quoting=csv.QUOTE_NONNUMERIC)


