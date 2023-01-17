# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:24:07 2022

Author: Rounak Meyur
Description: Plots the results of stability studies
"""

import sys, os
import matplotlib.pyplot as plt
import numpy as np


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNM = FLAT_NORM_MEAN = "\\widehat{{\\mathbb{{F}}}}_{{\\lambda}}"
FNI = FLAT_NORM_INDEX = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
HAUS = HAUSDORFF_DISTANCE = "\\mathbb{{D}}_{{Haus}}"
HAUSM = HAUSDORFF_DISTANCE_MEAN = "\\widehat{{\\mathbb{{D}}}}_{{Haus}}"

workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from libs.pyFlatNormFixture import FlatNormFixture, close_fig


# get fixture
fx = FlatNormFixture('runTest')
fx.out_dir = "out/test"
fx.fig_dir = "figs/test"
fx.area = "mcbryde"



# read geometries
area = 'mcbryde'

# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    # 967: 'ex3',
    # 930: 'ex4'
    }
radii = [10, 20, 30, 40, 50]

df_fn, df_ind = fx.read_stability_stats(
    f"{area}-FN_STABILITY_STAT_N100_R5",
    f"{area}-FN_STAT_INDEX",
    in_dir=fx.out_dir)



L = len(radii)


for i,index in enumerate(ind_label):
    fig, axs = plt.subplots(1, L, figsize=(L * 20, 20), constrained_layout=True)
    fn_means, hd_means, fn_index, hd_index = list(), list(), list(), list()
    
    for e, radius in enumerate(radii):
        fnm, hdm, fni, hdi = fx.plot_stability_result(
            df_fn, df_ind, index, radius=radius, ax=axs[e],
            title_fontsize=40, xylabel_fontsize=45, tick_fontsize=45,
            scatter_size=300, index_size=800,
        )
        fn_means.append(fnm)
        hd_means.append(hdm)
        fn_index.append(fni)
        hd_index.append(hdi)
        

    fnm_mean, fnm_std = np.array(fn_means).mean(), np.array(fn_means).std()
    hdm_mean, hdm_std = np.array(hd_means).mean(), np.array(hd_means).std()
    fni_mean, fni_std = np.array(fn_index).mean(), np.array(fn_index).std()
    hdi_mean, hdi_std = np.array(hd_index).mean(), np.array(hd_index).std()

    fnm_suptitle = f"${{\\sf mean}}({FNM})={fnm_mean:0.3g}, {{\\sf sd}}({FNM})={fnm_std:0.3g}$"
    fnm_suptitle_short = f"perturbed ${FNM}={fnm_mean:0.3g} \\pm {fnm_std:0.3g}$ ${HAUSM}={hdm_mean:0.3g} \\pm {hdm_std:0.3g}$"
    fnm_prefix = f"Scatter plot of ${FNN}$ for perturbed networks in region {ind_label[index]}"
    city_suptitle = f"original ${FNN}={fni_mean:0.3g}$, ${HAUS}={hdi_mean:0.3g}$"
    fig.suptitle(f"{fnm_prefix}  :  {city_suptitle}  :  {fnm_suptitle_short}", fontsize=70)

    file_name = f"{fx.area}-stability_flatnorm_{ind_label[index]}"
    close_fig(fig, to_file=f"{fx.fig_dir}/{file_name}.png", show=True)
