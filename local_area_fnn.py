# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:34:45 2022

@author: rm5nz
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

from libs.pyFlatNormFixture import FlatNormFixture, close_fig, get_fig_from_ax


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
fx.fig_dir = "figs/areas"
fx.out_dir = "out/areas"
fx.area = 'patrick_henry'
# fx.area = 'mcbryde'

num_regions = 50
lambda_ = 1000



flatnorm_df, fn_city_df, city_ratio = fx.read_stats(
    f"{fx.area}-FN_STAT_R{num_regions}-fixed_lambdas",
    f"{fx.area}-FN_STAT_city-fixed_lambdas",
    in_dir="out/test"
)

#%% Plot the location statistics

fig, axd = plt.subplot_mosaic(
    [
        ['scatter', 'scatter', 'scatter', 'city', 'city', 'city'],
        ['t11', 'r11', 't12', 'r12', 't13', 'r13'],
        ['t21', 'r21', 't22', 'r22', 't23', 'r23'],
        ['t31', 'r31', 't32', 'r32', 't33', 'r33'],
    ],
    figsize=(24, 24), constrained_layout=True,
    # width_ratios=[2, 2, 2, 2, 2],
    gridspec_kw={'wspace': 0.05,
                 'width_ratios':[2, 2, 2, 2, 2, 2]}
)


idx = pd.IndexSlice
fn_df_lambda = flatnorm_df.loc[idx[:, lambda_], ].copy()


max_regions_ids = fn_df_lambda \
    .sort_values(by='flatnorms', ascending=False)['id'] \
    .to_list()[:3]


city_ratio = fn_city_df['input_ratios'].max()
fn_df_lambda['to_city'] = abs(fn_df_lambda['input_ratios'] - city_ratio)
near_regions_ids = fn_df_lambda \
    .sort_values(by='to_city', ascending=True)['id'] \
    .to_list()[:3]

min_regions_ids = fn_df_lambda \
        .sort_values(by='flatnorms', ascending=True)['id'] \
        .to_list()[:3]

# Plot the regions
all_regions_ids = min_regions_ids + max_regions_ids + near_regions_ids

colors = ['xkcd:aqua', 'xkcd:violet', 'xkcd:tan']
markers = ['D','^','P']
leglabels = [f"Minimum ${FNN}$", f"Maximum ${FNN}$", 
             f"$(|T|/\\epsilon,{FNN})$ near $(|T_G|/\\epsilon_G,{FNC})$"]
leghandles = [Line2D([], [], marker=markers[i], 
                     label=leglabels[i], markerfacecolor=colors[i],
                     color='white') for i in range(len(colors))]

fx.plot_selected_regions(
    all_regions_ids, flatnorm_df, fn_city_df, lambda_,
    axd=axd,
    axtr=[axd['t11'], axd['t12'], axd['t13'],
          axd['t21'], axd['t22'], axd['t23'],
          axd['t31'], axd['t32'], axd['t33'],],
    axfn=[axd['r11'], axd['r12'], axd['r13'],
          axd['r21'], axd['r22'], axd['r23'],
          axd['r31'], axd['r32'], axd['r33'],],
    highlight_size=16**2, city_size=18**2,
    region_alpha=0.6,
    highlight_color=np.repeat(colors, 3), 
    highlight_marker=np.repeat(markers, 3),
    highlight_legend = leghandles,
    # reg_line=True, titles=['fn', 'corr', 'r2'], fontsize=25,
    reg_line=False, titles=['fn'], fontsize=30, xylabel_fontsize=25
)

fname = "noregline"

prefix = "Flat norm computation for local regions in"
city_suptitle = f"${CITY(fx.area)}$ : $\\lambda = {lambda_:d}$"
# fig.suptitle(f"{city_suptitle}", fontsize=35)

file_name = f"{fx.area}-R{num_regions}-{fname}"

close_fig(fig, to_file=f"{fx.fig_dir}/{file_name}.png", show=True)


