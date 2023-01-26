# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Chooses particular regions in a geographic location to compare the
local networks inside it. Perform perturbations to few node locations and address
aspect of metric stability with respect to outliers.
"""

import sys, os
import numpy as np
from shapely.geometry import Point, LineString
import pandas as pd
import csv
from tqdm.auto import tqdm


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
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

# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    # 967: 'ex3',
    # 930: 'ex4'
    }


# Perturbations

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

# compute flat norm
act_geom, synth_geom, hull = fx.read_networks(area)
struct = get_structure(act_geom)

# parameters
num_networks = 1000
radius_list = [10, 20, 30, 40, 50]
epsilon = 1e-3
lambda_ = 1e-3


# compute stats
flatnorm_stability_data = {
    'radius': [], 'flatnorms': [], 'input_ratios': [], 'index': [], 'hausdorff': [],
    'MIN_X': [], 'MIN_Y': [], 'MAX_X': [], 'MAX_Y': []
}

# Construct perturbed synthetic network
np.random.seed(123)
for radius in radius_list:
    
    # compute flat norm and append to statistics disctionary
    for ind in ind_label:
        point = Point(struct["vertices"][ind])
        region = fx.get_region(point, epsilon)
        sgeom_list = variant_geometry(synth_geom, region, radius=radius, N=num_networks)
        
        # Compute local flat norm for perturbed networks
        for i in tqdm(range(len(sgeom_list)), 
            desc="Multiple perturbations with outliers",
            ncols = 100, position=0, leave=True
            ):
            norm, hd, w = fx.compute_region_metric(
                act_geom, sgeom_list[i], point, epsilon, lambda_,
                plot = False,
                verbose=False, normalized = True,
                distance="geodesic" 
            )
            
            # Update statistics
            flatnorm_stability_data['radius'].append(radius)
            flatnorm_stability_data['index'].append(ind)
            flatnorm_stability_data['flatnorms'].append(norm)
            flatnorm_stability_data['input_ratios'].append(w/epsilon)
            flatnorm_stability_data['hausdorff'].append(hd)
            region_bounds = region.exterior.bounds
            flatnorm_stability_data['MIN_X'].append(region_bounds[MIN_X])
            flatnorm_stability_data['MIN_Y'].append(region_bounds[MIN_Y])
            flatnorm_stability_data['MAX_X'].append(region_bounds[MAX_X])
            flatnorm_stability_data['MAX_Y'].append(region_bounds[MAX_Y])



df_stability = pd.DataFrame(flatnorm_stability_data)

print("--------------------------------------------------------------------------")
print(
    f"compute flatnorm for {num_networks} perturbed networks for {len(ind_label)} local regions"
    )
print("--------------------------------------------------------------------------")



file_name = f"{area}-L{lambda_}_FN_STABILITY_STAT_OUTLIER_N{num_networks}_R{len(radius_list)}"

with open(f"{fx.out_dir}/{file_name}.csv", "w") as outfile:
    df_stability.to_csv(outfile, sep=",", index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

