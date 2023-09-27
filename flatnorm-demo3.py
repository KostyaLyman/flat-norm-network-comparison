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
#%%
from __future__ import absolute_import

import sys,os
import shapely.geometry as sg


workpath = os.getcwd()

from libs.pyFlatNormFixture import FlatNormFixture, get_fig_from_ax, close_fig


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
# for lambda_ in [1000, 50000, 100000]:
#     norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
#         D = D, T1=T1, T2=T2,
#         lambda_=lambda_,
#         normalized=False,
#         plot=True,
#         verbose=False,
#         opts="psVe"
#     )


#     # plot city flat norm
#     R = 6378
#     norm = norm * R

#     fig, ax = fx.plot_triangulated_region_flatnorm(
#         epsilon=epsilon, lambda_=lambda_, w=w, 
#         fnorm=norm,
#         show_figs = ["fn"],
#         to_file = f"{fx.area}-{lambda_}-flatnorm_city",
#         suptitle = f"Scale, $\\lambda$ = {lambda_}  :  "
#                     f"Flat norm, ${FN}$ = {norm:0.3f} km",
#         do_return=True,
#         constrained_layout=True,
#         figsize=(28,16), fontsize=75,
#         **plot_data
#     )




    
#%%



lambda_ = 1000
norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
        D = D, T1=T1, T2=T2,
        lambda_=lambda_,
        normalized=False,
        plot=True,
        verbose=False,
        opts="psVe"
    )

triangulated = plot_data.get("triangulated")
echain = plot_data.get("echain")
tchain = plot_data.get("tchain")
region = plot_data.get("region")


#%%
from libs.pyDrawNetworklib import plot_triangulation, plot_norm
fig, axs, no_ax = get_fig_from_ax(ax=None, ndim=(3,1), figsize=(30,50))

plot_triangulation(triangulated, T1, T2, axs[0], 
                    show_triangulation=False, 
                    region_bound=region, 
                    legend=True, location="lower right")

plot_triangulation(triangulated, T1, T2, axs[1], 
                    show_triangulation=True, 
                    region_bound=region, 
                    legend=True, location="lower right")

plot_norm(triangulated, echain, tchain, axs[2], 
            region_bound=region)

to_file = f"{fx.fig_dir}/{fx.area}-all-plot.png"
close_fig(fig, to_file, show=True, bbox_inches='tight')
# %%
