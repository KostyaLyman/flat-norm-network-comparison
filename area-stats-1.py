# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:00:19 2022

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
fx.fig_dir = "figs/test"
fx.out_dir = "out/test"



fx.area = 'patrick_henry'
lambdas = np.array([1, 25, 50, 75, 100], dtype=int) * 1000
num_regions = 50


flatnorm_df, fn_city_df, city_ratio = fx.read_stats(
    f"{fx.area}-FN_STAT_R{num_regions}-fixed_lambdas",
    f"{fx.area}-FN_STAT_city-fixed_lambdas",
    in_dir="out/test"
)

L = len(lambdas)
fig, axs = plt.subplots(1, L, figsize=(L * 15, 15), constrained_layout=True)

fn_means, fn_cities = list(), list()
for l, lambda_ in enumerate(lambdas):
    fnm, fnc = fx.plot_hist_fn(
        flatnorm_df, fn_city_df, lambda_, ax=axs[l],
        titles=['fn', 'city'], 
        title_fontsize=50, xtick_fontsize=50

    )
    fn_means.append(fnm)
    fn_cities.append(fnc)
    pass

fnm_mean, fnm_std = np.array(fn_means).mean(), np.array(fn_means).std()
fnc_mean, fnc_std = np.array(fn_cities).mean(), np.array(fn_cities).std()

# fnm_suptitle = f"${{\\sf mean}}({FNM})={fnm_mean:0.3g}, {{\\sf sd}}({FNM})={fnm_std:0.3g}$"
fnm_suptitle_short = f"${FNM}={fnm_mean:0.3g} \\pm {fnm_std:0.3g}$"
fnm_prefix = f"Histogram of ${FNN}$ for ${CITY(fx.area)}$"
city_suptitle = f"${FNC}={fnc_mean:0.3g} \\pm {fnc_std:0.3g}$"
fig.suptitle(f"{fnm_prefix}  :  {city_suptitle}  :  {fnm_suptitle_short}", fontsize=70)

file_name = f"{fx.area}-R{num_regions}-FIXL-flatnorm_hists_lambdas"
close_fig(fig, to_file=f"{fx.fig_dir}/{file_name}.png", show=True)

pass