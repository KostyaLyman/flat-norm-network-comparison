# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Chooses particular regions in a geographic location to compare the
networks inside it.
"""

import sys, os
from shapely.geometry import Point
import pandas as pd
import csv


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
fx.out_dir = "out/test"
fx.fig_dir = "figs/test"
fx.area = 'mcbryde'


# read geometries
act_geom, synt_geom, hull = fx.read_networks()
struct = get_structure(act_geom)


# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    967: 'ex3',
    930: 'ex4'
    }


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
    region_ID = ind_label[ind]
    region = fx.get_region(pt, epsilon)
    norm, hd, w, plot_data = fx.compute_region_metric(
        act_geom, synt_geom,
        pt, epsilon, lambda_,
        plot = True, distance="geodesic",
        normalized = True, verbose=False
        )
    
    fx.plot_triangulated_region_flatnorm(
        show_figs=["haus","fn"], 
        show=False, do_return=False,
        to_file=f"{fx.area}-L{lambda_}_fn_region_{region_ID}",
        legend_location = "upper left",
        **plot_data, 
        )

    # store the data
    flatnorm_data['index'].append(ind)
    flatnorm_data['flatnorms'].append(norm)
    flatnorm_data['input_ratios'].append(w/epsilon)
    flatnorm_data['hausdorff'].append(hd)
    region_bounds = region.exterior.bounds
    flatnorm_data['MIN_X'].append(region_bounds[MIN_X])
    flatnorm_data['MIN_Y'].append(region_bounds[MIN_Y])
    flatnorm_data['MAX_X'].append(region_bounds[MAX_X])
    flatnorm_data['MAX_Y'].append(region_bounds[MAX_Y])
    

df = pd.DataFrame(flatnorm_data)
filename = f"{fx.area}-L{lambda_}_FN_STAT_INDEX"
with open(f"{fx.out_dir}/{filename}.csv", "w") as outfile:
    df.to_csv(outfile, sep=",", index=False,
              header=True, quoting=csv.QUOTE_NONNUMERIC)

