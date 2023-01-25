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
    1003:'ex2',
    # 967: 'ex3',
    # 930: 'ex4'
    }

#%% Flat norm computation
def compute_metric_region(
    act_geom, synt_geom,
    ind, point, eps, lamb_, 
    plot_result=True, 
    suptitle_prefix="",
    show_plot=True,
    ):
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
    
    # compute hausdorff distance
    hd, hd_geom = fx.compute_region_hausdorff(
        fx.get_region(point, eps),
        act_geom, synt_geom,
        distance = "geodesic",
        )
    plot_data["hd_geom"] = hd_geom
    
    
    titles = {
        'lambda': f"$\\lambda = {lambda_:0.4f}$",
        'epsilon': f"$\\epsilon = {epsilon:0.4f}$",
        'ratio': f"$|T|/\\epsilon = {w/epsilon :0.3g}$",
        'fn': f"${FNN}={norm:0.3g}$",
        'haus': f"${HAUS}={hd:0.3g}$"
    }
    title = ", ".join([titles[t_name] for t_name in titles])
    
    # plot flat norm
    if plot_result:
        fx.plot_triangulated_region_flatnorm(
            epsilon=eps, lambda_=lamb_,
            to_file=f"{area}-L{lamb_}_outlier_region_{region}_{suptitle_prefix}",
            suptitle=f"{suptitle_prefix} : {title}",
            show_figs = ["haus", "fn"],
            do_return=False, show=show_plot,
            figsize=(26,14),
            legend_location = "upper left",
            **plot_data
            )
    return norm, hd, w

#%% Perturbation functions
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


# compute stats
flatnorm_data = {
    'flatnorms': [], 'input_ratios': [], 'index': [], 'hausdorff': [],
    'MIN_X': [], 'MIN_Y': [], 'MAX_X': [], 'MAX_Y': []
}

for ind in ind_label:
    pt = Point(struct["vertices"][ind])
    norm, hd, w = compute_metric_region(
        actual_geom, synthetic_geom,
        ind, pt, epsilon, lambda_, 
        suptitle_prefix = "",
        plot_result=True, show_plot=False
        )
    
    sgeom_list = variant_geometry(synthetic_geom, radius=2000, N=1)
    for l, syn_geom in enumerate(sgeom_list):
        norm, hd, w = compute_metric_region(
            actual_geom, syn_geom,
            ind, pt, epsilon, lambda_, 
            suptitle_prefix = f"perturbed{l+1}",
            plot_result=True, show_plot=False
            )
    


