# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:34:05 2022

@author: rm5nz
"""

import unittest

from timeit import default_timer as timer
from datetime import timedelta
from pprint import pprint
import warnings

import sys, os
import numpy as np
import math

import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import shapely.geometry as sg
from matplotlib import pyplot as plt
import matplotlib.axes
import networkx as nx
import pandas as pd
import seaborn as sns

from libs.pyExtractDatalib import GetDistNet
from libs.pyFlatNormlib import get_geometry, get_current, msfn, perform_triangulation
from libs.pyDrawNetworklib import plot_norm, plot_intermediate_result, plot_input, plot_failed_triangulation
from libs.pyDrawNetworklib import plot_regions, plot_triangulation


MIN_X, MIN_Y, MAX_X, MAX_Y = 0, 1, 2, 3
long_dashed = (5, (10, 3))

FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
# FNN = NORMALIZED_FLAT_NORM = "\\tilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNNM = FNM = FLAT_NORM_MEAN = "\\hat{{\\mathbb{{F}}}}_{{\\lambda}}"
FNNC = FNC = FLAT_NORM_CITY = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}^{{\\  G}}"
# CITY = lambda x: f"{{\\bf {x}}}"
CITY = lambda x: "{{\\bf {city}}}".format(city=x.replace("_", " ").title())


def get_fig_from_ax(ax, **kwargs):
    if not ax:
        no_ax = True
        ndim = kwargs.get('ndim', (1, 1))
        figsize = kwargs.get('figsize', (10, 10))
        constrained_layout = kwargs.get('constrained_layout', False)
        fig, ax = plt.subplots(*ndim, figsize=figsize, constrained_layout=constrained_layout)
    else:
        no_ax = False
        if not isinstance(ax, matplotlib.axes.Axes):
            if isinstance(ax, list):
                getter = kwargs.get('ax_getter', lambda x: x[0])
                ax = getter(ax)
            if isinstance(ax, dict):
                getter = kwargs.get('ax_getter', lambda x: next(iter(ax.values())))
                ax = getter(ax)
        fig = ax.get_figure()

    return fig, ax, no_ax


def close_fig(fig, to_file=None, show=True, **kwargs):
    if to_file:
        fig.savefig(to_file, **kwargs)
    if show:
        plt.show()
    plt.close(fig)
    pass


def timeit(f, *args, **kwargs):
    start = timer()
    outs = f(*args, **kwargs)
    end = timer()
    return outs, end - start


class FlatNormFixture(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.act_path = "./input/actual"
        self.synt_path = "./input/primnet"
        self._out_dir = "out"
        self._fig_dir = "figs"

        self.area_codes = {
            'mcbryde': [150692, 150724],
            'hethwood': [150692, 150724],
            'north_blacksburg': [150724, 150723, 150692],
            'patrick_henry': [150724, 150723, 150692],
        }
        pass

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out):
        self._out_dir = out
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        pass

    @property
    def fig_dir(self):
        return self._fig_dir

    @fig_dir.setter
    def fig_dir(self, fig):
        self._fig_dir = fig
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        pass

    @staticmethod
    def sort_geometry(input_geometry):
        output_geometry = []
        for geom in input_geometry:
            cord1 = geom.coords[0]
            cord2 = geom.coords[1]
            if cord1[0] > cord2[0]:
                output_geometry.append(LineString((Point(cord2), Point(cord1))))
            else:
                output_geometry.append(geom)
        return output_geometry

    def read_actual_network(self, area=None):
        if not area:
            area = self.area

        # Actual network
        act_edges_file = f"{self.act_path}/{area}/{area}_edges.shp"
        if not os.path.exists(act_edges_file):
            raise ValueError(f"{act_edges_file} doesn't exist!")

        df_act = gpd.read_file(act_edges_file)
        act_geom = []
        for i in range(len(df_act)):
            act_geom.extend(get_geometry(df_act['geometry'][i]))

        # Get convex hull of the region
        # act_lines = MultiLineString(act_geom)
        hull = MultiLineString(act_geom).convex_hull.buffer(5e-4)

        return act_geom, hull

    def read_synthetic_network(self, codes=list(), area=None, hull=None):
        if not area:
            area = self.area
        # Synthetic network
        if not codes:
            codes = self.area_codes[area]

        # synt_net = GetDistNet(self.synt_path, self.area_codes[area])
        synt_net = GetDistNet(self.synt_path, codes)

        # Get the synthetic network edges in the region
        if hull:
            synt_nodes = [
                n for n in synt_net.nodes
                if Point(synt_net.nodes[n]['cord']).within(hull)
                   and synt_net.nodes[n]['label'] != 'H'
            ]
            synt_net = nx.subgraph(synt_net, synt_nodes)

        # synt_geom = []
        # for e in synt_graph.edges:
        #     synt_geom.extend(get_geometry(synt_net.edges[e]['geometry']))
        synt_geom = [
            g
            # get_geometry(synt_net.edges[e]['geometry'])
            for e in synt_net.edges
            for g in get_geometry(synt_net.edges[e]['geometry'])
        ]

        return synt_geom

    def read_networks(self, area=None):
        if not area:
            area = self.area

        # # Actual network
        act_geom, hull = self.read_actual_network(area)
        # act_edges_file = f"{self.act_path}/{area}/{area}_edges.shp"
        # if not os.path.exists(act_edges_file):
        #     raise ValueError(f"{act_edges_file} doesn't exist!")
        #
        # df_act = gpd.read_file(act_edges_file)
        # act_geom = []
        # for i in range(len(df_act)):
        #     act_geom.extend(get_geometry(df_act['geometry'][i]))
        #
        # # Get convex hull of the region
        # act_lines = MultiLineString(act_geom)
        # hull = act_lines.convex_hull.buffer(5e-4)

        # Synthetic network
        # synt_net = GetDistNet(self.synt_path, self.area_codes[area])
        #
        # # Get the synthetic network edges in the region
        # synt_nodes = [
        #     n for n in synt_net.nodes
        #     if Point(synt_net.nodes[n]['cord']).within(hull)
        #        and synt_net.nodes[n]['label'] != 'H'
        # ]
        # synt_graph = nx.subgraph(synt_net, synt_nodes)
        # synt_geom = []
        # for e in synt_graph.edges:
        #     synt_geom.extend(get_geometry(synt_net.edges[e]['geometry']))

        synt_geom = self.read_synthetic_network(area=area, hull=hull)

        return act_geom, synt_geom, hull

    def read_stats(self, fn_stat_file, fn_city_file, in_dir=None):
        """
        :param fn_stat_file: f"{fx.area}-FN_STAT_R{num_regions}"
        :param fn_city_file: f"{fx.area}-FN_STAT_city"
        :return: fn_df, city_df, city_ratio
        """
        if in_dir:
            fn_stat_file = f"{in_dir}/{fn_stat_file}"
            fn_city_file = f"{in_dir}/{fn_city_file}"

        fn_stat_file = fn_stat_file if fn_stat_file.endswith(".csv") else f"{fn_stat_file}.csv"
        fn_city_file = fn_city_file if fn_city_file.endswith(".csv") else f"{fn_city_file}.csv"

        fn_stat_df = pd.read_csv(
            fn_stat_file,
            sep=",",
            dtype={
                'epsilons': float,
                'lambdas': int,
                'flatnorms': np.float64,
                'norm_lengths': np.float64,
                'norm_areas': np.float64,
                'input_lengths': np.float64,
                'input_ratios': np.float64,
                'MIN_X': np.float64,
                'MIN_Y': np.float64,
                'MAX_X': np.float64,
                'MAX_Y': np.float64,
            }
        )
        fn_stat_df['id'] = list(range(len(fn_stat_df)))
        fn_stat_df = fn_stat_df.set_index(['epsilons', 'lambdas'], drop=False)

        fn_city_df = pd.read_csv(
            fn_city_file,
            sep=",",
        )
        fn_city_df = fn_city_df.set_index(['lambdas'], drop=False)
        city_ratio = fn_city_df['input_ratios'].max()

        return fn_stat_df, fn_city_df, city_ratio

    @staticmethod
    def get_region(point, epsilon=2e-3):
        return point.buffer(epsilon, cap_style=sg.CAP_STYLE.square)

    @staticmethod
    def random_points(poly, num_points=5):
        min_x, min_y, max_x, max_y = poly.bounds
        points = []
        while len(points) < num_points:
            random_point = Point([np.random.uniform(min_x, max_x),
                                  np.random.uniform(min_y, max_y)])
            if (random_point.within(poly)):
                points.append(random_point)
        return points

    @staticmethod
    def random_points_buffered(poly, epsilon=2e-3, tollerance=0.1, num_points=5):
        min_x, min_y, max_x, max_y = poly.bounds
        points = []
        while len(points) < num_points:
            random_point = Point([np.random.uniform(min_x, max_x),
                                  np.random.uniform(min_y, max_y)])
            region = random_point.buffer(epsilon, cap_style=sg.CAP_STYLE.square)
            if random_point.within(poly) and region.intersection(poly).area / region.area > tollerance:
                points.append(random_point)
        return points

    @staticmethod
    def random_points_geom(poly, geom1, geom2, epsilon=2e-3, num_points=5):
        geom1 = MultiLineString(geom1)
        geom2 = MultiLineString(geom2)
        min_x, min_y, max_x, max_y = poly.bounds
        points = []
        while len(points) < num_points:
            # print(f"SAMPLED :: {len(points)} points")
            random_point = Point([np.random.uniform(min_x, max_x),
                                  np.random.uniform(min_y, max_y)])
            region = random_point.buffer(epsilon, cap_style=sg.CAP_STYLE.square)
            if random_point.within(poly):
                if geom1.intersects(region) and geom2.intersects(region):
                    points.append(random_point)

        return points

    def sample_regions(
            self, hull,
            num_regions=5, epsilon=2e-3,
            regions_only=False, seed=None
    ):
        if seed:
            np.random.seed(seed)
        point_list = self.random_points(hull, num_regions)
        region_list = [
            pt.buffer(epsilon, cap_style=sg.CAP_STYLE.square) for pt in point_list
        ]
        if regions_only:
            return region_list

        return point_list, region_list

    def sample_regions_buffered(
            self, hull,
            num_regions=5, epsilon=2e-3, tollerance=0.1,
            regions_only=False, seed=None
    ):
        if seed:
            np.random.seed(seed)
        point_list = self.random_points_buffered(hull, epsilon, tollerance, num_regions)
        region_list = [
            pt.buffer(epsilon, cap_style=sg.CAP_STYLE.square) for pt in point_list
        ]
        if regions_only:
            return region_list

        return point_list, region_list

    def sample_regions_geom(
            self, hull, act_geom, synt_geom,
            num_regions=5, epsilon=2e-3,
            regions_only=False, seed=None
    ):
        if seed:
            np.random.seed(seed)
        point_list = self.random_points_geom(hull, act_geom, synt_geom, epsilon,  num_regions)
        region_list = [
            pt.buffer(epsilon, cap_style=sg.CAP_STYLE.square) for pt in point_list
        ]
        if regions_only:
            return region_list

        return point_list, region_list

    def get_triangulated_currents(
            self, region, act_geom, synt_geom, **kwargs
    ):
        verbose = kwargs.get('verbose', False)
        adj = kwargs.get('adj', 1000)
        opts = kwargs.get('opts', "ps")

        # get the actual network edges in the region
        reg_act_geom = [g for g in act_geom if g.intersects(region)]
        reg_synt_geom = [g for g in synt_geom if g.intersects(region)]
        sorted_act_geom = self.sort_geometry(reg_act_geom)
        sorted_synt_geom = self.sort_geometry(reg_synt_geom)
        if verbose:
            print(f"Task completed: Actual[{len(sorted_act_geom)}] "
                  f"and synthetic[{len(sorted_synt_geom)}] network geometries sorted")

        # if not sorted_act_geom or not sorted_synt_geom:
        # if len(sorted_act_geom) == 0 or len(sorted_synt_geom) == 0:
        if len(sorted_act_geom) + len(sorted_synt_geom) == 0:
            return dict(), np.array([]), np.array([])

        # Flat norm computation
        D = perform_triangulation(
            sorted_act_geom, sorted_synt_geom,
            adj=adj, opts=opts,
            verbose=verbose,
        )
        if D['triangulated'] == None:
            _ = plot_failed_triangulation(D)
            sys.exit(0)

        # computing currents
        T1 = get_current(D['triangulated'], D['actual'])
        T2 = get_current(D['triangulated'], D['synthetic'])
        if verbose:
            print(f"Task completed: Actual[{abs(T1).sum()}] "
                  f"and synthetic[{abs(T2).sum()}] currents created")

        return D, T1, T2

    def compute_region_flatnorm(
            self, region=None, act_geom=None, synt_geom=None,
            D=None, T1=None, T2=None,
            lambda_=1000,
            normalized=False,
            **kwargs
            # verbose=False, plot=False, old_impl=False
    ):
        """
        :returns:  flatnorm, enorm, tnorm, input_length, [plot_data if plot==True]
        """
        verbose = kwargs.setdefault('verbose', False)
        plot = kwargs.get('plot', False)
        old_impl = kwargs.get('old_impl', False)

        # compute triangulation and currents
        # if not D or not T1 or not T2:
        if not D:
            D, T1, T2 = self.get_triangulated_currents(region, act_geom, synt_geom, **kwargs)
        T = T1 - T2

        if not D or T1.size == 0 or T2.size == 0:
            warnings.warn(f"D={len(D)} : T1={T1.size} : T2={T2.size}")
            return None, None, None, None

        # --- Multiscale flat norm ---
        # x: e-chain, s: t-chain,
        # norm: flat norm, enorm: weight of e-chain, tnorm: weight of t-chain,
        # w: vector of edges length of triangulation
        x, s, norm, enorm, tnorm, w = msfn(
            D['triangulated']['vertices'], D['triangulated']['triangles'], D['triangulated']['edges'],
            input_current=T,
            lambda_=lambda_,
            k=np.pi / (180.0),
            normalized=False
        )

        # --- normalization ---
        input_length = np.dot(abs(T1), abs(w)) + np.dot(abs(T2), abs(w))
        # if old_impl:
        #     input_length = np.dot(abs(T), abs(w))   # old implementation

        if normalized:
            norm = norm / input_length

        if verbose:
            print("The computed simplicial flat norm is:", norm)

        if plot:
            plot_data = dict(
                triangulated=D["triangulated"],
                T1=T1,
                T2=T2,
                echain=x,
                tchain=s,
            )
            return norm, enorm, tnorm, input_length, plot_data

        return norm, enorm, tnorm, input_length

    def plot_regions_list(
            self,
            act_geom, synt_geom, regions_list, area=None,
            # region_highlight=None,
            ax=None, to_file=None, show=True,
            **kwargs
    ):
        kwargs.setdefault('figsize', (40, 20))
        fontsize = kwargs.get('fontsize', 20)
        do_return = kwargs.get('do_return', False)
        if not area:
            area = self.area

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        plot_regions(act_geom, synt_geom, regions_list, ax, **kwargs)

        # title = f"[{area}]"
        title = f"${CITY(self.area)}$"
        if title_sfx := kwargs.get('title_sfx'):
            title = f"{title} : {title_sfx}"
        ax.set_title(title, fontsize=fontsize)

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}-regions"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{to_file}"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass

    def plot_triangulated_region_flatnorm(
            self, triangulated, T1, T2,
            echain, tchain,
            epsilon, lambda_,
            fnorm=None, fnorm_only=False,
            ax=None, to_file=None, show=True,
            **kwargs
    ):
        kwargs.setdefault('figsize', (20, 10))
        fontsize = kwargs.get('fontsize', 20)
        do_return = kwargs.get('do_return', False)

        # ---- PLOT ----
        fig, axs, no_ax = get_fig_from_ax(ax, ndim=(1, 2), **kwargs)
        fnorm_title = f"Flat norm scale, $\\lambda$ = {lambda_:d}" if not fnorm \
            else f"$\\lambda = {lambda_:d}$, ${FNN}={fnorm:0.3g}$"
            # else f"$\\lambda = {lambda_:d}$, $F_{{\\lambda}}={fnorm:0.3g}$"

        if not fnorm_only:
            plot_triangulation(triangulated, T1, T2, axs[0])
            axs[0].set_title(f"$\\epsilon = {epsilon}$", fontsize=fontsize)

            plot_norm(triangulated, echain, tchain, axs[1])
            axs[1].set_title(fnorm_title, fontsize=fontsize)
        else:
            plot_norm(triangulated, echain, tchain, axs)
            axs.set_title(fnorm_title, fontsize=fontsize)

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{to_file}"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize + 3)
            close_fig(fig, to_file, show, bbox_inches='tight')

        if do_return:
            return fig, axs
        pass

    def plot_region_flatnorm_lines(
            self, epsilons, lambdas, flatnorms,
            ax=None, to_file=None, show=True,
            **kwargs
    ):
        kwargs.setdefault('figsize', (10, 10))
        fontsize = kwargs.get('fontsize', 16)
        fontsize = kwargs.get('xylabel_fontsize', fontsize)
        title_fontsize = kwargs.get('title_fontsize', fontsize + 3)

        do_return = kwargs.get('do_return', False)

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        ax = sns.lineplot(
            # data=plot_data_df,
            x=lambdas, y=flatnorms, hue=epsilons, ax=ax,
            markers='o',
            palette=sns.color_palette("mako_r", 16)
        )
        ax.set_xlabel(f"$\lambda$", fontsize=fontsize)
        ax.set_ylabel(f"${FNN}$",
                      rotation=kwargs.get('y_label_rotation', 'vertical'),
                      fontsize=fontsize)

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{to_file}"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx};"
            fig.suptitle(suptitle, fontsize=title_fontsize)
            close_fig(fig, to_file, show, bbox_inches='tight')

        if do_return:
            return fig, ax
        pass

    def plot_hist_fn(self, fn_df, city_df, lambda_, epsilon=None, ax=None, **kwargs):
        fig, ax, _ = get_fig_from_ax(ax, **kwargs)
        lambda_ = int(lambda_)

        colors = {
            'city': kwargs.get('city_color', 'xkcd:electric blue'),
            'hist': kwargs.get('hist_color', 'xkcd:pastel blue'),
            'fn_mean': kwargs.get('fn_mean_color', 'xkcd:kelly green'),
        }

        # flatnorm VS ratios ------------------------------------------------------
        if epsilon:
            fn_df = fn_df.loc[(epsilon, lambda_),].copy()
        else:
            idx = pd.IndexSlice
            fn_df = fn_df.loc[idx[:, lambda_],].copy()

        # flatnorm histogram -------------------------------------------------------
        hist_data = ax.hist(
            fn_df['flatnorms'],
            bins=20, color=colors['hist'],
            range=(0, 1),
        )

        # mean flatnorm -----------------------------------------------------------
        fn_mean = fn_df['flatnorms'].mean()
        fn_sd = fn_df['flatnorms'].std()
        ax.plot(
            [fn_mean, fn_mean], [0, 100],
            c=colors['fn_mean'], linewidth=3
        )
        ax.plot(
            [fn_mean - fn_sd, fn_mean - fn_sd], [0, 100],
            c=colors['fn_mean'], linewidth=1, linestyle=long_dashed
        )
        ax.plot(
            [fn_mean + fn_sd, fn_mean + fn_sd], [0, 100],
            c=colors['fn_mean'], linewidth=1, linestyle=long_dashed
        )

        # city flatnorm -----------------------------------------------------------
        city_fn = city_df.at[lambda_, 'flatnorms']
        ax.plot(
            [city_fn, city_fn], [0, 100],
            c=colors['city'], linewidth=3
        )

        # ticks -------------------------------------------------------------------
        ax.set_xticks([city_fn, fn_mean],
                      labels=[f"{city_fn:0.3g}", f"{fn_mean:0.3g}"],
                      minor=True,
                      )

        for tlabel, tcolor in zip(ax.get_xticklabels(minor=True), [colors['city'], colors['fn_mean']]):
            tlabel.set_color(tcolor)

        # plot title --------------------------------------------------------------
        titles = dict(
            # fn=f"$\\hat{{F}}_{{\\lambda}}={fn_mean:0.3f} \pm {fn_sd:0.3f}$",
            fn=f"${FNM} = {fn_mean:0.3g} \pm {fn_sd:0.3g}$",
            city=f"${FNC} = {city_fn:0.3g}$"
        )
        which_titles = kwargs.get('titles', list(titles.keys()))

        title_styles = dict(
            ll=f"$\\lambda={{lambda_:d}}$",
            le=f"$\\lambda={{lambda_:d}}$, $\\epsilon={{epsilon:0.4f}}$",
            el=f"$\\epsilon={{epsilon:0.4f}}$, $\\lambda={{lambda_:d}}$",
        )
        title_style = kwargs.get('title_style', 'le')

        if epsilon:
            title = title_styles[title_style].format(lambda_=lambda_, epsilon=epsilon)
        else:
            title = title_styles['ll'].format(lambda_=lambda_)

        # subtitle = ", ".join([t_str for t_name, t_str in titles.items() if t_name in which_titles])
        subtitle = ", ".join([titles[t_name] for t_name in which_titles])
        title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(title, fontsize=kwargs.get('title_fontsize', 18))

        # plot config -------------------------------------------------------------
        ax.set_xlabel(f"${FNN}$",
                      # rotation='horizontal',
                      fontsize=kwargs.get('xylabel_fontsize', 16))

        ax.set_ylim(bottom=0, top=hist_data[0].max() + 1)
        ax.set_xlim(left=-0.02, right=1.02)
        return fn_mean, city_fn

    def plot_fn_vs_ratio(self, fn_df, city_df, lambda_, epsilon=None, ax=None, **kwargs):
        """
        :param kwargs: highlight: set, highlight_marker: str;
                       highlight_size: int, scatter_size: int, city_size: int;
                       titles: list, title_style: str;
                       reg_line: bool, mean_line: bool, city_point: bool;
                       y_label_rotation: str;
                       X_color: str -- where X in {city, scatter, highlight, fn_mean, regression}
                       X_fontsize: int -- where X in {title, xylabel};
                       if ax is None: figsize, constrained_layout
        """
        fig, ax, _ = get_fig_from_ax(ax, **kwargs)
        lambda_ = int(lambda_)

        colors = {
            'city': kwargs.get('city_color', 'xkcd:electric blue'),
            'scatter': kwargs.get('scatter_color', 'xkcd:pastel blue'),
            'highlight': kwargs.get('highlight_color', 'xkcd:pumpkin'),
            'fn_mean': kwargs.get('fn_mean_color', 'xkcd:kelly green'),
            'regression': kwargs.get('regression_color', 'red'),
        }

        # flatnorm VS ratios ------------------------------------------------------
        if epsilon:
            fn_df = fn_df.loc[(epsilon, lambda_), ].copy()
        else:
            idx = pd.IndexSlice
            fn_df = fn_df.loc[idx[:, lambda_], ].copy()

        x_ratios = fn_df['input_ratios'].to_numpy()
        y_flatnorms = fn_df['flatnorms'].to_numpy()

        if highlight := kwargs.get('highlight', list()):
            if not isinstance(highlight, (list, set)):
                highlight = set(highlight)
            fn_df['color'] = colors['scatter']
            fn_df.loc[fn_df['id'].isin(highlight), 'color'] = colors['highlight']
            colors['scatter'] = fn_df['color'].to_list()

        ax.scatter(
            x_ratios, y_flatnorms,
            alpha=0.7,
            s=kwargs.get('scatter_size', 5 ** 2),
            c=colors['scatter'],
            marker='o'
        )

        if highlight:
            ax.scatter(
                fn_df.loc[fn_df['id'].isin(highlight), 'input_ratios'],
                fn_df.loc[fn_df['id'].isin(highlight), 'flatnorms'],
                alpha=0.8,
                c=colors['highlight'],
                s=kwargs.get('highlight_size', 12 ** 2),
                marker=kwargs.get('highlight_marker', 'x'),
            )

        # mean line ---------------------------------------------------------------
        fn_mean = y_flatnorms.mean()
        fn_sd = y_flatnorms.std()
        if kwargs.get('mean_line', True):
            ax.plot(
                [0, 100], [fn_mean, fn_mean],
                c=colors['fn_mean'], linewidth=3, alpha=0.5
            )
            ax.plot(
                [0, 100], [fn_mean - fn_sd, fn_mean - fn_sd],
                c=colors['fn_mean'], linewidth=1, linestyle=long_dashed, alpha=0.5,
            )
            ax.plot(
                [0, 100], [fn_mean + fn_sd, fn_mean + fn_sd],
                c=colors['fn_mean'], linewidth=1, linestyle=long_dashed, alpha=0.5,
            )

        # city flatnorm -----------------------------------------------------------
        city_fn, city_ratio = city_df.loc[lambda_, ['flatnorms', 'input_ratios']]
        if kwargs.get('city_point', True):
            ax.scatter(
                [city_ratio], [city_fn],
                alpha=1,
                s=kwargs.get('city_size', 15 ** 2),
                color=colors['city'],
                marker='*'
            )

        # regression line ---------------------------------------------------------
        from sklearn.metrics import r2_score
        # regression
        N = len(x_ratios)
        b, a = np.polyfit(x_ratios, y_flatnorms, deg=1)
        y_predict = a + b * x_ratios
        y_var = sum((y_predict - y_flatnorms) ** 2) / (N - 1)
        y_err = math.sqrt(y_var)
        b_var = y_var / sum((x_ratios - x_ratios.mean()) ** 2)
        b_err = math.sqrt(b_var)
        # correlation & R2
        fn_ratio_corr = np.corrcoef(x_ratios, y_flatnorms)[0, 1]
        fn_ratio_r2 = r2_score(y_flatnorms, y_predict)

        if kwargs.get('reg_line', True):
            xseq = np.linspace(0, 1, num=10)
            ax.plot(xseq, a + b * xseq, color=colors['regression'], lw=3, alpha=0.5)
            ax.plot(
                xseq, a + y_err + b * xseq,
                color=colors['regression'], lw=1, alpha=0.5, linestyle=long_dashed
            )
            ax.plot(
                xseq, a - y_err + b * xseq,
                color=colors['regression'], lw=1, alpha=0.5, linestyle=long_dashed
            )

        # stat data dict ----------------------------------------------------------
        stat_dict = {
            'a': a, 'b': b,
            'err': math.sqrt(y_var),
            'std_err': np.std(y_predict - y_flatnorms),
            'b_err': b_err,
            'corr': fn_ratio_corr,
            'R2': fn_ratio_r2,
            'fn_mean': fn_mean,
            'fn_sd': fn_sd,
        }

        # ticks -------------------------------------------------------------------
        ax.set_xticks([city_ratio], [f"{city_ratio:0.3g}"], color=colors['city'], 
                      minor=True, )
        ax.set_yticks([city_fn, fn_mean],
                      labels=[f"{city_fn:0.3g}", f"{fn_mean:0.3g}"],
                      minor=True,
                      )

        for tlabel, tcolor in zip(ax.get_yticklabels(minor=True), [colors['city'], colors['fn_mean']]):
            tlabel.set_color(tcolor)

        # plot title --------------------------------------------------------------
        titles = dict(
            fn=f"${FNM}={fn_mean:0.3g} \pm {fn_sd:0.3g}$",
            beta=f"$\\hat{{\\beta}}={b:0.3g} \pm {b_err:0.3g}$",
            corr=f"$\\rho={fn_ratio_corr:0.3g}$",
            r2=f"$R^{{2}}={fn_ratio_r2:0.3g}$",
        )
        which_titles = kwargs.get('titles', list(titles.keys()))

        title_styles = dict(
            ll=f"$\\lambda={{lambda_:d}}$",
            le=f"$\\lambda={{lambda_:d}}$, $\\epsilon={{epsilon:0.4f}}$",
            el=f"$\\epsilon={{epsilon:0.4f}}$, $\\lambda={{lambda_:d}}$",
        )
        title_style = kwargs.get('title_style', 'le')

        if epsilon:
            title = title_styles[title_style].format(lambda_=lambda_, epsilon=epsilon)
        else:
            title = title_styles['ll'].format(lambda_=lambda_)

        subtitle = ", ".join([titles[t_name] for t_name in which_titles])
        title = f"{title}\n{subtitle}"
        ax.set_title(title, fontsize=kwargs.get('title_fontsize', 25))

        # plot config -------------------------------------------------------------
        ax.set_xlabel("$Ratio |T|/\\epsilon$", 
                      fontsize=kwargs.get('xylabel_fontsize', 20))
        ax.set_ylabel(f"Normalized flat norm ${FNN}$",
                      rotation=kwargs.get('y_label_rotation', 'vertical'),
                      fontsize=kwargs.get('xylabel_fontsize', 20))
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1.05)
        return stat_dict

    def plot_selected_regions(self,
                              region_ids, fn_df, city_df, lambda_, epsilon=None,
                              axd=dict(), axfn=list(), axtr=list(),
                              **kwargs):
        """
        :param kwargs:
               plot_fn_vs_ratio ::
                        highlight: set, titles: list, title_style: str, X_color: str, X_fontsize: int;
                        if ax is None: figsize, constrained_layout
        """
        kwargs.setdefault('titles', ['fn', 'beta', 'r2'])
        kwargs.setdefault('title_style', "ll")
        fontsize = kwargs.setdefault('fontsize', 20)
        fontsize = kwargs.setdefault('title_fontsize', fontsize)

        colors = {
            'highlight': kwargs.setdefault('highlight_color', 'xkcd:pumpkin'),
        }

        # PLOT SETUP --------------------------------------------------------------------
        if not axd:
            fig, axd = plt.subplot_mosaic(
                [
                    ['scatter', 'city'],
                ],
                figsize=kwargs.get('figsize', (12, 6)), constrained_layout=True,
                width_ratios=[3, 7],
                gridspec_kw={'wspace': 0.05}
            )
            axfn = list()
            axtr = list()

        # PLOT SCATTER ------------------------------------------------------------------
        stats_data = self.plot_fn_vs_ratio(
            fn_df, city_df, lambda_, epsilon=epsilon,
            ax=axd['scatter'],
            highlight=region_ids,
            **kwargs
        )

        # PLOT REGIONS AND CITY ---------------------------------------------------------
        act_geom, synt_geom, hull = kwargs.get('geometries', (None, None, None))
        if act_geom is None:
            act_geom, synt_geom, hull = self.read_networks()

        selected_regions = dict()
        fn_df = fn_df.set_index('id')
        for r_id in region_ids:
            region_bounds = fn_df.loc[r_id, ['MIN_X', 'MIN_Y', 'MAX_X', 'MAX_Y']].to_list()
            selected_regions[r_id] = sg.box(*region_bounds)

        city_epsilon, city_fn, city_ratio = city_df.loc[lambda_, ['epsilons', 'flatnorms', 'input_ratios']]
        self.plot_regions_list(
            act_geom, synt_geom, list(selected_regions.values()), self.area,
            ax=axd['city'],
            region_color=colors['highlight'],
            title_sfx=f"$\\epsilon_{{G}} = {city_epsilon:0.4g} : "
                      f"|T_{{G}}|/\\epsilon_{{G}} = {city_ratio:0.3g} : "
                      f"{FNC} = {city_fn:0.3g}$",
            do_return=False, show=False,
            **kwargs
        )

        # PLOT SELECTED REGIONS ---------------------------------------------------------
        if axfn or axtr:
            which_titles = kwargs.get('region_titles', ['epsilon', 'ratio', 'fn'])
            for r, r_id in enumerate(region_ids):
                # flat norm
                epsilon, ratio, fnorm = fn_df.loc[r_id, ['epsilons', 'input_ratios', 'flatnorms']]
                region = selected_regions[r_id]
                titles = {
                    'lambda': f"$\\lambda = {lambda_:d}$",
                    'epsilon': f"$\\epsilon = {epsilon}$",
                    'ratio': f"$|T|/\\epsilon = {ratio:0.3g}$",
                    'fn': f"${FNN}={fnorm:0.3g}$",
                }
                title = ", ".join([titles[t_name] for t_name in which_titles])

                D, T1, T2 = self.get_triangulated_currents(region, act_geom, synt_geom, opts="ps")
                
                if axtr and not axfn:
                    which_titles = kwargs.get('region_titles', ['epsilon', 'ratio', 'fn'])
                    title = ", ".join([titles[t_name] for t_name in which_titles])
                    plot_triangulation(D["triangulated"], T1, T2, axtr[r])
                    axtr[r].set_title(title, fontsize=fontsize)
                else:
                    if axtr:
                        which_titles = kwargs.get('region_titles', ['epsilon', 'ratio'])
                        title = ", ".join([titles[t_name] for t_name in which_titles])
                        plot_triangulation(D["triangulated"], T1, T2, axtr[r])
                        axtr[r].set_title(title, fontsize=fontsize)
    
                    if axfn:
                        which_titles = kwargs.get('region_titles', ['fn'])
                        title = ", ".join([titles[t_name] for t_name in which_titles])
                        fnorm, enorm, tnorm, w, plot_data = self.compute_region_flatnorm(
                            D=D, T1=T1, T2=T2,
                            lambda_=lambda_,
                            normalized=True,
                            plot=True
                        )
                        plot_norm(plot_data['triangulated'], plot_data['echain'], plot_data['tchain'], axfn[r])
                        axfn[r].set_title(title, fontsize=fontsize)

        return stats_data

        # pass
