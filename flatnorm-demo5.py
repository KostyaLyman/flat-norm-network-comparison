# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Displays impact of perturbation on the nodes of a network
"""

import sys, os
from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd
import csv


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
HAUS = HAUSDORFF_DISTANCE = "\\mathbb{{D}}_{{Haus}}"
MIN_X, MIN_Y, MAX_X, MAX_Y = 0, 1, 2, 3


workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from libs.pyFlatNormFixture import FlatNormFixture, get_fig_from_ax, close_fig
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
    # 1003:'ex2',
    # 967: 'ex3',
    # 930: 'ex4'
    }



# Perturbation functions
def get_perturbed_verts(vertices, radius, perturb_index=None):
    n = vertices.shape[0]
    # Get the deviation radius in degrees
    R = 6378100
    phi = (180/np.pi) * (radius/R)
    
    # Sample vertices from the radius of existing vertices
    r = phi * np.sqrt(np.random.uniform(size=(n,)))
    theta = np.random.uniform(size=(n,)) * 2 * np.pi
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    # Selectively perturb particular vertices
    if not perturb_index:
        perturb = np.ones(shape=(n,))
    else:
        perturb = np.zeros(shape=(n,))
        perturb[perturb_index] = 1
    
    return vertices + (np.diag(perturb) @ np.vstack((dx,dy)).T)

def variant_geometry(geometry, region, radius=10, N=1):
    # Get the vertices and line segments
    struct = get_structure(geometry)
    
    # Get the vertices within region
    vert_ind = [i for i,p in enumerate(struct["vertices"]) if Point(p).within(region)]

    new_geom = []
    for i in range(N):
        # Get perturbed vertices
        perturb_index = np.random.choice(vert_ind, size=(1,))
        
        new_verts = get_perturbed_verts(
            struct["vertices"], radius, 
            perturb_index=perturb_index
            )
    
        # get the updated geometries
        new_geom.append([LineString((Point(new_verts[i]), Point(new_verts[j]))) \
                    for i,j in struct["segments"]])
    return new_geom


epsilon = 1e-3
lambda_ = 1e-3
ind = 994
num_networks = 1
rad = 20


# for ind in ind_label:
pt = Point(act_struct["vertices"][ind])
region_ID = ind_label[ind]
region = fx.get_region(pt, epsilon)

# original without perturbation
norm, hd, w, plot_data = fx.compute_region_metric(
    actual_geom, synthetic_geom,
    pt, epsilon, lambda_,
    plot = True, distance="geodesic",
    normalized = True, verbose=False
    )

all_plot_data = [plot_data]

sgeom_list = variant_geometry(synthetic_geom, region, radius=rad, N=num_networks)

for i, synt_geom in enumerate(sgeom_list):
    norm, hd, w, plot_data = fx.compute_region_metric(
        actual_geom, synt_geom,
        pt, epsilon, lambda_,
        plot = True, distance="geodesic",
        normalized = True, verbose=False
        )

    all_plot_data.append(plot_data)

# Generate the plot
fx.plot_multiple_triangulated_region_flatnorm( 
    all_plot_data, show_figs = ["haus", "fn"], 
    to_file = f"{fx.area}-L{lambda_}_outlier_N{num_networks}_radius{rad}", 
    suptitle = f"Perturbed networks with outliers with a maximum displacement of {rad} meters",
    show = True,
    figsize=(32,32)
)


