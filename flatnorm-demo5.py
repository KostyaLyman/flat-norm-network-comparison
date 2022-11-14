# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Chooses particular regions in a geographic location to compare the
local networks inside it. Perform perturbations to the node locations and address
aspect of metric stability.
"""

import sys, os
import numpy as np
from shapely.geometry import Point, LineString
import pandas as pd


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
MIN_X, MIN_Y, MAX_X, MAX_Y = 0, 1, 2, 3


workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from pyFlatNormFixture import FlatNormFixture
from pyFlatNormlib import get_structure


# get fixture
fx = FlatNormFixture('runTest')
fx.out_dir = "out/test"
fx.fig_dir = "figs/test"



# read geometries
area = 'mcbryde'
act_geom, synth_geom, hull = fx.read_networks(area)
struct = get_structure(act_geom)


# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    967: 'ex3',
    930: 'ex4'}

#%% Perturbations

def get_perturbed_verts(vertices, radius):
    n = vertices.shape[0]
    # Get the deviation radius in degrees
    R = 6378100
    phi = (180/np.pi) * (radius/R)
    
    # Sample vertices from the radius of existing vertices
    r = phi * np.sqrt(np.random.uniform(size=(n,)))
    theta = np.random.uniform(size=(n,)) * 2 * np.pi
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return vertices + np.vstack((dx,dy)).T

def variant_geometry(geometry, radius=10, N=1):
    # Get the vertices and line segments
    struct = get_structure(geometry)
    
    # Get the new networks
    new_geom = []
    for i in range(N):
        # Get perturbed vertices
        new_verts = get_perturbed_verts(struct["vertices"], radius)
    
        # get the updated geometries
        new_geom.append([LineString((Point(new_verts[i]), Point(new_verts[j]))) \
                    for i,j in struct["segments"]])
    return new_geom

#%% compute flat norm

# parameters
num_networks = 100
radius_list = [10,20,30,40,50]
epsilon = 1e-3
lambda_ = 1000


# compute stats
flatnorm_stability_data = {
    'radius':[], 'flatnorms': [], 'input_ratios': [], 'index': [],
    'MIN_X': [], 'MIN_Y': [], 'MAX_X': [], 'MAX_Y': []
}

# Construct perturbed synthetic network
for rad in radius_list:
    sgeom_list = variant_geometry(synth_geom, radius=rad, N=num_networks)
    # compute flat norm and append to statistics disctionary
    for ind in ind_label:
        point = Point(struct["vertices"][ind])
        region = fx.get_region(point, epsilon)
        
        # Compute local flat norm for perturbed networks
        for syn_geom in sgeom_list:
            norm, enorm, tnorm, w = fx.compute_region_flatnorm(
                region,
                act_geom, syn_geom,
                lambda_=lambda_,
                normalized=True,
                plot=False
            )
            
            # Update statistics
            flatnorm_stability_data['radius'].append(rad)
            flatnorm_stability_data['index'].append(ind)
            flatnorm_stability_data['flatnorms'].append(norm)
            flatnorm_stability_data['input_ratios'].append(w/epsilon)
            region_bounds = region.exterior.bounds
            flatnorm_stability_data['MIN_X'].append(region_bounds[MIN_X])
            flatnorm_stability_data['MIN_Y'].append(region_bounds[MIN_Y])
            flatnorm_stability_data['MAX_X'].append(region_bounds[MAX_X])
            flatnorm_stability_data['MAX_Y'].append(region_bounds[MAX_Y])



df_stability = pd.DataFrame(flatnorm_stability_data)

print("--------------------------------------------------------------------------")
print(
    f"compute flatnorm for {num_networks} perturbed networks"
    f"for {len(radius_list)} radii "
    # f"and {len(lambdas)} lambdas = {timedelta(seconds=end_global - start_global)}"
    )
print("--------------------------------------------------------------------------")



file_name = f"{area}-FN_STABILITY_STAT_N{num_networks}_R{len(radius_list)}"
import csv
with open(f"{fx.out_dir}/{file_name}.csv", "w") as outfile:
    df_stability.to_csv(outfile, sep=",", index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

