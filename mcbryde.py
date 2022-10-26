# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:34:45 2022

@author: rm5nz
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

from pyFlatNormFixture import FlatNormFixture, close_fig, get_fig_from_ax


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
# FNN = NORMALIZED_FLAT_NORM = "\\tilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNNM = FNM = FLAT_NORM_MEAN = "\\widebar{{\\mathbb{{F}}}}_{{\\lambda}}"
FNNC = FNC = FLAT_NORM_CITY = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}^{{\\  C}}"
# CITY = lambda x: f"{{\\bf {x}}}"
CITY = lambda x: "{{\\bf {city}}}".format(city=x.replace("_", " ").title())


fx = FlatNormFixture('runTest')
fx.fig_dir = "figs/mcbryde"
fx.out_dir = "out/mcbryde"
fx.area = 'mcbryde'



fx.area = 'mcbryde'
lambdas = np.array([1, 25, 50, 75, 100], dtype=int) * 1000
num_regions = 50


flatnorm_df, fn_city_df, city_ratio = fx.read_stats(
    f"{fx.area}-FN_STAT_R{num_regions}-fixed_lambdas",
    f"{fx.area}-FN_STAT_city-fixed_lambdas",
    in_dir="out/test"
)

L = len(lambdas)

fig, axd = plt.subplot_mosaic(
    [
        ['scatter', 'scatter', 'city', 'city', 'city'],
        ['t1', 't2', 't3', 't4', 't5'],
        ['r1', 'r2', 'r3', 'r4', 'r5'],
    ],
    figsize=(25, 16), constrained_layout=True,
    # width_ratios=[2, 2, 2, 2, 2],
    gridspec_kw={'wspace': 0.05,
                 'width_ratios':[2, 2, 2, 2, 2]}
)

l, lambda_ = 0, lambdas[0]

idx = pd.IndexSlice
fn_df_lambda = flatnorm_df.loc[idx[:, lambda_], ].copy()

selected_regions_ids = fn_df_lambda \
    .sort_values(by='flatnorms', ascending=False)['id'] \
    .to_list()[:5]

#%% Just plot the regions

fx.plot_selected_regions(
    selected_regions_ids, flatnorm_df, fn_city_df, lambda_,
    axd=axd,
    axtr=[axd['t1'], axd['t2'], axd['t3'], axd['t4'], axd['t5']],
    axfn=[axd['r1'], axd['r2'], axd['r3'], axd['r4'], axd['r5'], ],
    highlight_size=16**2, city_size=18**2,
    region_alpha=0.4,
    highlight_color='xkcd:violet',
    titles=['fn', 'corr', 'r2']
)

# city_suptitle = f"${CITY(self.area)} : |T|/\\epsilon = {city_ratio:0.3}$"
detail = "Local regions with maximum normalized flat norms"
city_suptitle = f"${CITY(fx.area)}$ : {detail}$\ {FNN}$ : $\\lambda = {lambda_:d}$"
fig.suptitle(f"{city_suptitle}", fontsize=25)

file_name = f"{fx.area}-R{num_regions}-FIXL-regions_l{lambda_}-max_fn"

close_fig(fig, to_file=f"{fx.fig_dir}/{file_name}.png", show=True)