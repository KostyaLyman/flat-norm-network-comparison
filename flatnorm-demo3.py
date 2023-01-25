# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:23:40 2022

Author: Rounak Meyur

Description: Demonstration of how varying the scale parameter of simplicial 
flat norm alters the computed norm. The test is done for actual and synthetic
distribution networks in Blacksburg. 

The norm is computed with the two components separately: subsimplicial norm and
simplicial norm. These data are stored in a .txt file for multiple scale values.
"""

from __future__ import absolute_import

import sys,os
import shapely.geometry as sg


workpath = os.getcwd()

from libs.pyFlatNormFixture import FlatNormFixture


area_name = {
    'mcbryde': r'Location\ A',
    'patrick_henry': r'Location\ B',
    'hethwood': r'Location\ C'}

FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNNM = FNM = FLAT_NORM_MEAN = "\\widehat{{\\mathbb{{F}}}}_{{\\lambda}}"
FNNC = FNC = FLAT_NORM_CITY = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}^{{\\  G}}"
CITY = lambda x: f"{{\\bf {area_name[x]}}}"




fx = FlatNormFixture('runTest')
fx.fig_dir = "figs/test"
fx.out_dir = "out/test"
fx.area = 'mcbryde'

# read geometries
act_geom, synt_geom, hull = fx.read_networks(fx.area)

# city region
MIN_X, MIN_Y, MAX_X, MAX_Y = 0, 1, 2, 3
city_bounds = hull.exterior.bounds
city_region = sg.box(*city_bounds)
city_width, city_height = city_bounds[MAX_X] - city_bounds[MIN_X], city_bounds[MAX_Y] - city_bounds[MIN_Y]
epsilon = max(city_width/2, city_height/2)


# entire location flat norm
D, T1, T2 = fx.get_triangulated_currents(city_region, act_geom, synt_geom)


#%% Run for different lambdas
for lambda_ in [1000, 50000, 100000][:1]:
    norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
        D = D, T1=T1, T2=T2,
        lambda_=lambda_,
        normalized=False,
        plot=True,
        verbose=False,
        opts="psVe"
    )


    # plot city flat norm
    R = 6378
    norm = norm * R
    fig, ax = fx.plot_triangulated_region_flatnorm(
        epsilon=f"{epsilon:0.4f}", lambda_=lambda_,
        fnorm_only=True,
        to_file = f"{fx.area}-{lambda_}-flatnorm_city",
        suptitle = f"Scale, $\\lambda$ = {lambda_}  :  "
                    f"Flat norm, ${FN}$ = {norm:0.3f} km",
        do_return=True,
        constrained_layout=True,
        figsize=(26,16), fontsize=60,
        **plot_data
    )










sys.exit(0)

#%% Flat norm computation for multiple lambda

import numpy as np
for lambda_ in np.linspace(1000,100000,100):
    
    # Plot results for every 10 computation
    if int(lambda_)%10000 == 0:
        norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
            D = D, T1=T1, T2=T2,
            lambda_=lambda_,
            normalized=True,
            plot=True,
            verbose=False,
            opts="psVe"
        )


        # plot city flat norm
        R = 6378
        norm = norm * R
        fig, ax = fx.plot_triangulated_region_flatnorm(
            epsilon=f"{epsilon:0.4f}", lambda_=lambda_,
            fnorm_only=True,
            to_file = f"{fx.area}-{lambda_}-flatnorm_city",
            suptitle = f"${FNN}$={norm:0.5f} : "
                         f"|T| / $\\epsilon$ = {w / epsilon:0.5f}",
            do_return=True,
            constrained_layout=True,
            figsize=(26,16), fontsize=30,
            **plot_data
        )
    else:
        norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
            D = D, T1=T1, T2=T2,
            lambda_=lambda_,
            normalized=True,
            plot=True,
            verbose=False,
            opts="psVe"
        )
    # Save the norm results
    with open(f"{fx.out_dir}flat-norm.txt",'a') as f:
        data = "\t".join([str(x) for x in [int(lambda_),norm,
                                            enorm,tnorm]]) + "\n"
        f.write(data)
    


