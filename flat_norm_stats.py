import sys, os
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString, base as sg
from matplotlib import pyplot as plt, axes
from timeit import default_timer as timer
from datetime import timedelta
from pprint import pprint


from flat_norm_fixture import FlatNormFixture
if __name__ == '__main__':
    # get fixture
    fx = FlatNormFixture('runTest')
    fx.seed = 54321
    fx.out_dir = "out/script"
    fx.fig_dir = "figs/script"

    # parameters
    area = 'mcbryde'
    epsilons, lambdas = np.linspace(5e-4, 2e-3, 4), np.linspace(1000, 100000, 5)
    num_regions = 50

    # read geometries
    act_geom, synt_geom, hull = fx.read_networks(area)

    # compute stats
    flatnorm_data = {
        'epsilons': [], 'lambdas': [], 'flatnorms': [],
        'norm_lengths': [], 'norm_areas': [],
        'input_lengths': [], 'input_ratios': [],
    }

    np.random.seed(fx.seed)
    start_global = timer()
    for e, epsilon in enumerate(epsilons, start=1):
        for l, lambda_ in enumerate(lambdas, start=1):
            print(f"### EPS[{e}] = {epsilon:0.5f} ###### LAMBDA[{l}] = {lambda_:0.5f} ###")

            # sample points
            points = fx.random_points_geom(
                hull, act_geom, synt_geom,
                epsilon=epsilons[0],
                num_points=num_regions,
            )

            # compute flat norm
            start = timer()
            for pt in points:
                norm, enorm, tnorm, w = fx.compute_region_flatnorm(
                    fx.get_region(pt, epsilon),
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


    file_name = f"{area}-flatnorm-stats_{num_regions}_regions"
    import csv
    with open(f"{fx.out_dir}/{file_name}.csv", "w") as outfile:
        flatnorm_data.to_csv(outfile, sep=",", index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
