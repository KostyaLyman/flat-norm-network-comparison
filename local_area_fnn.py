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

num_regions = 50
lambda_ = 1000

highlight_type_list = ["min","near","max"]


flatnorm_df, fn_city_df, city_ratio = fx.read_stats(
    f"{fx.area}-FN_STAT_R{num_regions}-fixed_lambdas",
    f"{fx.area}-FN_STAT_city-fixed_lambdas",
    in_dir="out/test"
)

#%% Plot the location statistics
for highlight_type in highlight_type_list:
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
    
    
    idx = pd.IndexSlice
    fn_df_lambda = flatnorm_df.loc[idx[:, lambda_], ].copy()
    
    
    if highlight_type == "max":
        selected_regions_ids = fn_df_lambda \
            .sort_values(by='flatnorms', ascending=False)['id'] \
            .to_list()[:5]
        detail = f"Local regions with high normalized flat norm $\ {FNN}$"
        fname = "max_fn"
    
    elif highlight_type == "near":
        city_ratio = fn_city_df['input_ratios'].max()
        fn_df_lambda['to_city'] = abs(fn_df_lambda['input_ratios'] - city_ratio)
        selected_regions_ids = fn_df_lambda \
            .sort_values(by='to_city', ascending=True)['id'] \
            .to_list()[:5]
        detail = f"$(|T|/\\epsilon, {FNN})$ values close to global value $(|T_{{G}}|/\\epsilon_{{G}}, {FNC})$"
        fname = "near_fn"
    
    elif highlight_type == "min":
        selected_regions_ids = fn_df_lambda \
                .sort_values(by='flatnorms', ascending=True)['id'] \
                .to_list()[:5]
        detail = f"Local regions with small normalized flat norm $\ {FNN}$"
        fname = "min_fn"
    
    else:
        raise ValueError(f"Invalid highlight type {highlight_type}")
        sys.exit(0)
    
    # Plot the regions
    fx.plot_selected_regions(
        selected_regions_ids, flatnorm_df, fn_city_df, lambda_,
        axd=axd,
        axtr=[axd['t1'], axd['t2'], axd['t3'], axd['t4'], axd['t5']],
        axfn=[axd['r1'], axd['r2'], axd['r3'], axd['r4'], axd['r5'], ],
        highlight_size=16**2, city_size=18**2,
        region_alpha=0.4,
        highlight_color='xkcd:violet',
        reg_line=True, titles=['fn', 'corr', 'r2'], fontsize=25,
        # reg_line=False, titles=['fn'], fontsize=25,
    )
    
    # fname = fname + "_noregline"
    
    city_suptitle = f"${CITY(fx.area)}$ : {detail} : $\\lambda = {lambda_:d}$"
    fig.suptitle(f"{city_suptitle}", fontsize=35)
    
    file_name = f"{fx.area}-R{num_regions}-{fname}"
    
    close_fig(fig, to_file=f"{fx.fig_dir}/{file_name}.png", show=True)


