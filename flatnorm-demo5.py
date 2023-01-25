# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Chooses particular regions in a geographic location to compare the
networks inside it.
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
area = 'mcbryde'
actual_geom, synthetic_geom, hull = fx.read_networks(area)
struct = get_structure(actual_geom)


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
    r = phi
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

def variant_geometry(geometry, radius=10, N=1):
    # Get the vertices and line segments
    struct = get_structure(geometry)
    
    # Get the new networks
    n = struct["vertices"].shape[0]
    new_geom = []
    for i in range(N):
        # Get perturbed vertices
        perturb_index = np.random.randint(low=0, high=n, size=(1,))
        
        new_verts = get_perturbed_verts(
            struct["vertices"], radius, 
            perturb_index=perturb_index
            )
    
        # get the updated geometries
        new_geom.append([LineString((Point(new_verts[i]), Point(new_verts[j]))) \
                    for i,j in struct["segments"]])
    return new_geom

#%% compute flat norm

# parameters
epsilon = 1e-3
lambda_ = 1e-3


for ind in ind_label:
    pt = Point(struct["vertices"][ind])
    region_ID = ind_label[ind]
    region = fx.get_region(pt, epsilon)
    norm, hd, w = fx.compute_region_metric(
        actual_geom, synthetic_geom,
        pt, epsilon, lambda_, 
        plot_result=True, show = False, 
        show_figs = ["haus", "fn"], 
        to_file = f"{area}-L{lambda_}_fn_region_{region_ID}", 
        figsize=(26,14), legend_location = "upper left", 
        )
    
    sgeom_list = variant_geometry(synthetic_geom, radius=2000, N=1)
    
    


