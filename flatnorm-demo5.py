# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Displays impact of perturbation on the nodes of a network
"""

import sys, os
from shapely.geometry import Point, LineString
import numpy as np



FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
HAUS = HAUSDORFF_DISTANCE = "\\mathbb{{D}}_{{Haus}}"
MIN_X, MIN_Y, MAX_X, MAX_Y = 0, 1, 2, 3


workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from libs.pyFlatNormFixture import FlatNormFixture
from libs.pyFlatNormlib import get_structure


# get fixture
fx = FlatNormFixture('runTest')
fx.out_dir = "out/stability"
fx.fig_dir = "figs/stability"

# read geometries
fx.area = 'mcbryde'
actual_geom, synthetic_geom, hull = fx.read_networks()
act_struct = get_structure(actual_geom)


# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    # 967: 'ex3',
    # 930: 'ex4'
    }


epsilon = 1e-3
lambda_ = 1e-3
ind = 994



# for ind in ind_label:
pt = Point(act_struct["vertices"][ind])
region_ID = ind_label[ind]
region = fx.get_region(pt, epsilon)

# original without perturbation
norm, hd, w, org_plot_data = fx.compute_region_metric(
    actual_geom, synthetic_geom,
    pt, epsilon, lambda_,
    plot = True, distance="geodesic",
    normalized = True, verbose=False
    )





# create variant synthetic geometry
R = 6378100
phi = lambda rad: (180/np.pi) * (rad/R)

syn_struct = get_structure(synthetic_geom)

vert_ind = [i for i,p in enumerate(syn_struct["vertices"]) if Point(p).within(region)]
n = syn_struct["vertices"].shape[0]
perturb_index = vert_ind[3]

dx = np.zeros(shape=(n,))
dy = np.zeros(shape=(n,))
dx[perturb_index] = phi(-90)
dy[perturb_index] = phi(-170)
new_verts = syn_struct["vertices"] + np.vstack((dx,dy)).T

synt_geom = [LineString((Point(new_verts[i]), Point(new_verts[j]))) \
            for i,j in syn_struct["segments"]]

# compute the flatnorm and hausdorff distance 
norm, hd, w, pert_plot_data = fx.compute_region_metric(
    actual_geom, synt_geom,
    pt, epsilon, lambda_,
    plot = True, distance="geodesic",
    normalized = True, verbose=False
    )


# Generate the plot
fx.plot_multiple_triangulated_region_flatnorm( 
    org_plot_data, show_figs = ["haus", "fn"], 
    to_file = f"{fx.area}-L{lambda_}_original", 
    suptitle = "Original networks: Hausdorff and Flatnorm distances",
    fontsize=45,
    show = False,
    figsize=(28,15)
)

fx.plot_multiple_triangulated_region_flatnorm( 
    pert_plot_data, show_figs = ["haus", "fn"], 
    to_file = f"{fx.area}-L{lambda_}_perturbed", 
    suptitle = "Perturbed networks: Hausdorff and Flatnorm distances",
    fontsize=45,
    show = False,
    figsize=(28,15)
)