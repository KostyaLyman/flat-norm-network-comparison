import unittest

from timeit import default_timer as timer
from datetime import timedelta
from pprint import pprint
import warnings

import sys, os
import numpy as np

import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, base as sg
from matplotlib import pyplot as plt
import matplotlib.axes
import networkx as nx
import pandas as pd
import seaborn as sns

from libs.pyExtractDatalib import GetDistNet
from libs.pyFlatNormlib import get_geometry, get_current, msfn, perform_triangulation
from libs.pyDrawNetworklib import plot_norm, plot_intermediate_result, plot_input, plot_failed_triangulation
from libs.pyDrawNetworklib import plot_regions, plot_triangulation


def get_fig_from_ax(ax, figsize, **kwargs):
    if not ax:
        no_ax = True
        ndim = kwargs.get('ndim', (1, 1))
        fig, ax = plt.subplots(*ndim, figsize=figsize)
    else:
        no_ax = False
        if not isinstance(ax, matplotlib.axes.Axes):
            getter = kwargs.get('ax_getter', lambda x: x[0])
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

        self.areas = {
            'mcbryde': [150692, 150724],
            'hethwood': [150692, 150724],
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

    def read_networks(self, area):
        # Actual network
        act_edges_file = f"{self.act_path}/{area}/{area}_edges.shp"
        if not os.path.exists(act_edges_file):
            raise ValueError(f"{act_edges_file} doesn't exist!")

        df_act = gpd.read_file(act_edges_file)
        act_geom = []
        for i in range(len(df_act)):
            act_geom.extend(get_geometry(df_act['geometry'][i]))

        # Get convex hull of the region
        act_lines = MultiLineString(act_geom)
        hull = act_lines.convex_hull.buffer(5e-4)

        # Synthetic network
        synt_net = GetDistNet(self.synt_path, self.areas[area])

        # Get the synthetic network edges in the region
        synt_nodes = [
            n for n in synt_net.nodes
            if Point(synt_net.nodes[n]['cord']).within(hull)
               and synt_net.nodes[n]['label'] != 'H'
        ]
        synt_graph = nx.subgraph(synt_net, synt_nodes)
        synt_geom = []
        for e in synt_graph.edges:
            synt_geom.extend(get_geometry(synt_net.edges[e]['geometry']))

        return act_geom, synt_geom, hull

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
            self, region, act_geom, synt_geom, verbose=False,
    ):
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
        D = perform_triangulation(sorted_act_geom, sorted_synt_geom, adj=1000)
        if D['triangulated'] == None:
            _ = plot_failed_triangulation(D)
            sys.exit(0)

        # computing currents
        T1 = get_current(D['triangulated'], D['actual'])
        T2 = get_current(D['triangulated'], D['synthetic'])

        return D, T1, T2

    # def compute_single_flatnorm(
    #         self,
    #         D=None, T1=None, T2=None,
    #         region=None, act_geom=None, synt_geom=None,
    #         lambda_=1000,
    #         normalized=False,
    #         verbose=False,
    # ):
    #     if not D or not T1 or not T2:
    #         D, T1, T2 = self.get_triangulated_currents(region, act_geom, synt_geom, verbose=verbose)
    #     T = T1 - T2
    #
    #     if not D or T1.size == 0 or T2.size == 0:
    #         warnings.warn(f"D={len(D)} : T1={T1.size} : T2={T2.size}")
    #         return None, None, None, None
    #
    #     # Multiscale flat norm
    #     # x: e-chain, s: t-chain,
    #     # norm: flat norm, enorm: weight of e-chain, tnorm: weight of t-chain,
    #     # w: weight of input current T
    #     x, s, norm, enorm, tnorm, w = msfn(
    #         D['triangulated']['vertices'], D['triangulated']['triangles'], D['triangulated']['edges'],
    #         input_current=T,
    #         lambda_=lambda_,
    #         k=np.pi / (180.0),
    #         normalized=normalized
    #     )
    #     if verbose:
    #         print("The computed simplicial flat norm is:", norm)
    #
    #     return norm, enorm, tnorm, w

    def compute_region_flatnorm(
            self, region, act_geom, synt_geom,
            lambda_=1000,
            normalized=False,
            verbose=False, plot=False, old_impl=False
    ):
        # compute triangulation and currents
        D, T1, T2 = self.get_triangulated_currents(region, act_geom, synt_geom, verbose=verbose)
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
        if old_impl:
            input_length = np.dot(abs(T), abs(w))   # old implementation

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
            region_highlight=None,
            ax=None, to_file=None, show=True,
            **kwargs
    ):
        figsize = kwargs.get('figsize', (40, 60))
        do_return = kwargs.get('do_return', False)

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, figsize)
        plot_regions(act_geom, synt_geom, regions_list, ax, region_highlight)
        if area:
            to_file = f"{area}-regions"
            ax.set_title(f"[{area}]")

        if kwargs.get('file_name_sfx'):
            to_file = f"{to_file}_{kwargs.get('file_name_sfx')}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{to_file}"
            if kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {kwargs.get('suptitle_sfx')}"

            fig.suptitle(suptitle)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass

    def plot_triangulated_region_flatnorm(
            self, triangulated, T1, T2,
            echain, tchain,
            epsilon, lambda_,
            ax=None, to_file=None, show=True,
            **kwargs
    ):
        figsize = kwargs.get('figsize', (20, 10))
        fontsize = kwargs.get('fontsize', 20)
        do_return = kwargs.get('do_return', False)

        # ---- PLOT ----
        fig, axs, no_ax = get_fig_from_ax(ax, figsize, ndim=(1, 2))

        plot_triangulation(triangulated, T1, T2, axs[0])
        axs[0].set_title(f"Radius of region = {epsilon}", fontsize=fontsize)

        plot_norm(triangulated, echain, tchain, axs[1])
        axs[1].set_title(f"Flat norm scale, $\\lambda$ = {lambda_:d}", fontsize=fontsize)

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{to_file}"
            if kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {kwargs.get('suptitle_sfx')}"

            fig.suptitle(suptitle)
            close_fig(fig, to_file, show, bbox_inches='tight')

        if do_return:
            return fig, axs
        pass

    def plot_region_flatnorm_lines(
            self, epsilons, lambdas, flatnorms,
            ax=None, to_file=None, show=True,
            **kwargs
    ):
        figsize = kwargs.get('figsize', (10, 10))
        fontsize = kwargs.get('fontsize', 20)
        do_return = kwargs.get('do_return', False)

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, figsize)
        # colors = sns.color_palette(n_colors=16)
        ax = sns.lineplot(
            # data=plot_data_df,
            x=lambdas, y=flatnorms, hue=epsilons, ax=ax,
            markers='o',
            palette=sns.color_palette("mako_r", 16)
        )
        ax.set_xlabel(r"Scale ($\lambda$) for multi-scale flat norm", fontsize=fontsize)
        ax.set_ylabel(r"Normalized multi-scale flat norm", fontsize=fontsize)

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{to_file}"
            if kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} :  {kwargs.get('suptitle_sfx')};"
            fig.suptitle(suptitle)
            close_fig(fig, to_file, show, bbox_inches='tight')

        if do_return:
            return fig, ax
        pass


class FlatNormRuns(FlatNormFixture):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.seed = 54321
        pass

    def test_sample_regions(self):
        area = 'mcbryde'
        epsilon = 2e-3
        num_regions = 6

        # read geometries
        act_geom, synt_geom, hull = self.read_networks(area)
        self.assertIsNotNone(act_geom)
        self.assertIsNotNone(synt_geom)
        self.assertIsNotNone(hull)

        # sample regions
        point_list, region_list = self.sample_regions(
            hull, num_regions=num_regions, epsilon=epsilon, seed=self.seed
        )
        self.assertIsNotNone(region_list)

        # plot regions
        self.fig_dir = "figs/test"
        fig, ax = self.plot_regions_list(
            act_geom, synt_geom, region_list, area,
            do_return=True
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        pass

    def test_sample_regions_timed(self):
        area = 'mcbryde'
        epsilon = 2e-3
        num_regions = 6

        # read geometries
        (act_geom, synt_geom, hull), t_geom = timeit(
            self.read_networks, area
        )
        self.assertIsNotNone(act_geom)
        self.assertIsNotNone(synt_geom)
        self.assertIsNotNone(hull)

        # sample regions
        (point_list, region_list), t_regions = timeit(
            self.sample_regions,
            hull, num_regions=num_regions, epsilon=epsilon, seed=self.seed
        )
        self.assertIsNotNone(region_list)

        # plot regions
        self.fig_dir = "figs/test"
        fig, ax = self.plot_regions_list(
            act_geom, synt_geom, region_list, area, do_return=True
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        print(f"read geometries={timedelta(seconds=t_geom)}")
        print(f"sample regions={timedelta(seconds=t_regions)}")

        pass

    def test_compute_and_plot_region_flatnorm(self):
        area = 'mcbryde'
        epsilon, lambda_ = 2e-3, 1000
        num_regions = 6
        region = 2

        # read geometries
        act_geom, synt_geom, hull = self.read_networks(area)

        # sample regions
        point_list, region_list = self.sample_regions(
            hull, num_regions=num_regions, epsilon=epsilon, seed=self.seed
        )

        # plot regions
        self.fig_dir = "figs/test"
        self.plot_regions_list(
            act_geom, synt_geom, region_list, area,
            region_highlight=region,
            file_name_sfx=f"_region_{region}",
            do_return=False,
            figsize=(40, 60)
        )

        # flat norm
        norm, enorm, tnorm, w, plot_data = self.compute_region_flatnorm(
            region_list[region], act_geom, synt_geom,
            lambda_=lambda_,
            normalized=True,
            plot=True
        )
        self.assertIsNotNone(norm)
        self.assertIsNotNone(enorm)
        self.assertIsNotNone(tnorm)
        self.assertIsNotNone(w)
        self.assertIsNotNone(plot_data)

        # plot flat norm
        fig, ax = self.plot_triangulated_region_flatnorm(
            epsilon=epsilon, lambda_=lambda_,
            to_file=f"{area}-flatnorm_region_{region}",
            suptitle_sfx=f"FN={norm:0.5f} : |T| = {w:0.5f} : |T| / $\\epsilon$ = {w / epsilon:0.5f}",
            do_return=True,
            **plot_data
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        pass

    def test_compute_and_plot_region_flatnorm_old_impl(self):
        area = 'mcbryde'
        epsilon, lambda_ = 2e-3, 1000
        num_regions = 6
        region = 2

        # read geometries
        act_geom, synt_geom, hull = self.read_networks(area)

        # sample regions
        point_list, region_list = self.sample_regions(
            hull, num_regions=num_regions, epsilon=epsilon, seed=self.seed
        )

        # plot regions
        self.fig_dir = "figs/test"
        self.plot_regions_list(
            act_geom, synt_geom, region_list, area,
            region_highlight=region,
            file_name_sfx=f"_region_{region}",
            do_return=False,
            figsize=(40, 60)
        )

        # flat norm
        norm, enorm, tnorm, w, plot_data = self.compute_region_flatnorm(
            region_list[region], act_geom, synt_geom,
            lambda_=lambda_,
            normalized=True,
            plot=True
        )
        self.assertIsNotNone(norm)
        self.assertIsNotNone(enorm)
        self.assertIsNotNone(tnorm)
        self.assertIsNotNone(w)
        self.assertIsNotNone(plot_data)

        # plot flat norm
        fig, ax = self.plot_triangulated_region_flatnorm(
            epsilon=epsilon, lambda_=lambda_,
            to_file=f"{area}-flatnorm_region_{region}_old_impl",
            suptitle_sfx=f"OLD_IMPL : FN={norm:0.5f} : |T| = {w:0.5f} : |T| / $\\epsilon$ = {w / epsilon:0.5f}",
            do_return=True,
            **plot_data
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        pass

    def test_compute_and_plot_region_flatnorm_timed(self):
        area = 'mcbryde'
        epsilon, lambda_ = 2e-3, 1000
        num_regions = 6
        region = 2

        # read geometries
        (act_geom, synt_geom, hull), t_geom = timeit(
            self.read_networks, area
        )

        # sample regions
        (point_list, region_list), t_regions = timeit(
            self.sample_regions,
            hull, num_regions=num_regions, epsilon=epsilon, seed=self.seed
        )

        # plot regions
        self.fig_dir = "figs/test"
        _, t_regions_plot = timeit(
            self.plot_regions_list,
            act_geom, synt_geom, region_list, area,
            region_highlight=region,
            file_name_sfx=f"region_{region}",
            do_return=False,
            figsize=(40, 20)
        )

        # flat norm
        (norm, enorm, tnorm, w, plot_data), t_flatnorm = timeit(
            self.compute_region_flatnorm,
            region_list[region], act_geom, synt_geom,
            lambda_=lambda_,
            normalized=True,
            plot=True
        )
        self.assertIsNotNone(norm)
        self.assertIsNotNone(enorm)
        self.assertIsNotNone(tnorm)
        self.assertIsNotNone(w)
        self.assertIsNotNone(plot_data)
        self.assertIsNotNone(t_flatnorm)

        # plot flat norm
        self.fig_dir = "figs/test"
        (fig, ax), t_flatnorm_plot = timeit(
            self.plot_triangulated_region_flatnorm,
            epsilon=epsilon, lambda_=lambda_,
            to_file=f"{area}-flatnorm_region_{region}",
            suptitle_sfx=f"FN={norm:0.5f} : |T| = {w:0.5f} : |T| / $\\epsilon$ = {w / epsilon:0.5f}",
            do_return=True,
            **plot_data
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        print("--------------------------------------------------------------------------")
        print(f"read geometries={timedelta(seconds=t_geom)}")
        print(f"sample regions={timedelta(seconds=t_regions)}")
        print(f"plot regions={timedelta(seconds=t_regions_plot)}")
        print(f"flat norm for region={timedelta(seconds=t_flatnorm)}")
        print(f"plot flat norm for region={timedelta(seconds=t_flatnorm_plot)}")

        pass

    def test_plot_region_flatnorm_lines(self):
        area = 'mcbryde'
        # epsilons, lambdas = np.linspace(1e-3, 2e-3, 4), np.linspace(1000, 100000, 5)
        epsilons, lambdas = np.linspace(5e-4, 2e-3, 4), np.linspace(1000, 100000, 5)
        num_regions = 6
        region = 2

        # read geometries
        act_geom, synt_geom, hull = self.read_networks(area)

        # sample points
        # points, regions = self.sample_regions(
        #     hull, num_regions=num_regions, epsilon=epsilons[-1], seed=self.seed
        # )
        # points, regions = self.sample_regions_buffered(
        #     hull, num_regions=num_regions,
        #     epsilon=epsilons[0], tollerance=0.9,
        #     seed=12345
        # )
        points, regions = self.sample_regions_geom(
            hull, act_geom, synt_geom,
            epsilon=epsilons[0],
            num_regions=num_regions,
            seed=self.seed
        )

        # plot regions
        self.fig_dir = "figs/test"
        self.plot_regions_list(
            act_geom, synt_geom, regions, area,
            region_highlight=region,
            file_name_sfx=f"_region_{region}",
            do_return=False, show=True,
            figsize=(40, 20)
        )

        # flat norm
        flatnorm_data = {
            'epsilons': [], 'lambdas': [], 'flatnorms': [],
            'norm_lengths': [], 'norm_areas': []
        }

        start = timer()
        for epsilon in epsilons:
            for lambda_ in lambdas:
                norm, enorm, tnorm, w = self.compute_region_flatnorm(
                    self.get_region(points[region], epsilon),
                    act_geom, synt_geom,
                    lambda_=lambda_,
                    normalized=True,
                    plot=False
                )
                flatnorm_data['epsilons'].append(f"{epsilon:0.4f}")
                flatnorm_data['lambdas'].append(lambda_)
                flatnorm_data['flatnorms'].append(norm)
                flatnorm_data['norm_lengths'].append(enorm)
                flatnorm_data['norm_areas'].append(tnorm)

        end = timer()

        fig, ax = self.plot_region_flatnorm_lines(
            epsilons=flatnorm_data['epsilons'],
            lambdas=flatnorm_data['lambdas'],
            flatnorms=flatnorm_data['flatnorms'],
            to_file=f"{area}-flatnorm-lines_region_{region}",
            do_return=True
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        print("--------------------------------------------------------------------------")
        print(f"compute region flatnorm "
              f"for {len(epsilons)} epsilons "
              f"and {len(lambdas)} lambdas = {timedelta(seconds=end-start)}")
        print("--------------------------------------------------------------------------")
        pprint(pd.DataFrame(flatnorm_data))
        pass


    def test_flatnorm_stats(self):
        area = 'mcbryde'
        epsilons, lambdas = np.linspace(5e-4, 2e-3, 4), np.linspace(1000, 100000, 5)
        num_regions = 50
        # region = 2

        # read geometries
        act_geom, synt_geom, hull = self.read_networks(area)

        # sample points
        flatnorm_data = {
            'epsilons': [], 'lambdas': [], 'flatnorms': [],
            'norm_lengths': [], 'norm_areas': [],
            'input_lengths': [], 'input_ratios': [],
        }
        np.random.seed(self.seed)
        start_global = timer()
        for e, epsilon in enumerate(epsilons, start=1):
            for l, lambda_ in enumerate(lambdas, start=1):
                print(f"### EPS[{e}] = {epsilon:0.5f} ###### LAMBDA[{l}] = {lambda_:0.5f} ###")
                points = self.random_points_geom(
                    hull, act_geom, synt_geom,
                    epsilon=epsilons[0],
                    num_points=num_regions,
                )
                start = timer()
                for pt in points:
                    norm, enorm, tnorm, w = self.compute_region_flatnorm(
                        self.get_region(pt, epsilon),
                        act_geom, synt_geom,
                        lambda_=lambda_,
                        normalized=True,
                        plot=False
                    )
                    flatnorm_data['epsilons'].append(f"{epsilon:0.4f}")
                    flatnorm_data['lambdas'].append(lambda_)
                    flatnorm_data['flatnorms'].append(norm)
                    flatnorm_data['norm_lengths'].append(enorm)
                    flatnorm_data['norm_areas'].append(tnorm)
                    flatnorm_data['input_lengths'].append(w)
                    flatnorm_data['input_ratios'].append(w/epsilon)
                    pass

                end = timer()
                print(f">>> EPS[{e}] : LAMBDA[{l}] >>> {timedelta(seconds=end - start)} \n")


        end_global = timer()

        flatnorm_data = pd.DataFrame(flatnorm_data)

        print("--------------------------------------------------------------------------")
        print(
            f"compute flatnorm for {num_regions} regions"
            f"for {len(epsilons)} epsilons "
            f"and {len(lambdas)} lambdas = {timedelta(seconds=end_global - start_global)}")
        print("--------------------------------------------------------------------------")
        pprint(flatnorm_data)

        self.out_dir = "out/test"
        file_name = f"{area}-flatnorm-stats_{num_regions}_regions"
        import csv
        with open(f"{self.out_dir}/{file_name}.csv", "w") as outfile:
            flatnorm_data.to_csv(outfile, sep=",", index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
        pass

if __name__ == '__main__':
    unittest.main()
