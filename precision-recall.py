# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:23:40 2022

Author: Rounak Meyur
"""

from __future__ import absolute_import

import sys,os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import matplotlib.pyplot as plt
import networkx as nx


workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
actpath = workpath + "/input/actual/"
synpath = workpath + "/input/primnet/"
outpath = workpath + "/out/"

from pyExtractDatalib import GetDistNet
from pyFlatNormlib import get_geometry, get_current, msfn, perform_triangulation
from pyDrawNetworklib import plot_norm, plot_intermediate_result, plot_input, plot_failed_triangulation
from pyDrawNetworklib import plot_regions, plot_triangulation


#%% Functions
def random_regions(poly, num_regions = 5, epsilon = 2e-3):
    min_x, min_y, max_x, max_y = poly.bounds

    regions = []

    while len(regions) < num_regions:
        random_point = Point([np.random.uniform(min_x, max_x), 
                              np.random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            regions.append(random_point.buffer(epsilon,cap_style=3))

    return regions

def sort_geometry(input_geometry):
    output_geometry = []
    for geom in input_geometry:
        cord1 = geom.coords[0]
        cord2 = geom.coords[1]
        if cord1[0] > cord2[0]:
            output_geometry.append(LineString((Point(cord2),Point(cord1))))
        else:
            output_geometry.append(geom)
    return output_geometry

#%% Extract actual and synthetic network data
np.random.seed(12345)
area = 'mcbryde'
sublist = [150692,150724]

# Synthetic network
synth = GetDistNet(synpath,sublist)

# Actual network
df_lines = gpd.read_file(actpath+area+'/'+area+'_edges.shp')
actgeom = []
for i in range(len(df_lines)):
    actgeom.extend(get_geometry(df_lines['geometry'][i]))

# Get convex hull of the region
act_lines = MultiLineString(actgeom)
hull = act_lines.convex_hull.buffer(5e-4)

# Get the synthetic network edges in the region
nodelist = [n for n in synth.nodes \
            if Point(synth.nodes[n]['cord']).within(hull) \
                and synth.nodes[n]['label'] != 'H']
syn_graph = nx.subgraph(synth,nodelist)
syngeom = []
for e in syn_graph.edges:
    syngeom.extend(get_geometry(synth.edges[e]['geometry']))

# Sample small regions within the convex hull
region_list = random_regions(hull)

# Plot the sample regions
fig = plt.figure(figsize=(40,60))
ax = fig.add_subplot(111)
plot_regions(actgeom,syngeom,region_list,ax)
filename = area+'-regions'
fig.savefig(workpath+'/figs/'+filename+'.png',bbox_inches='tight')

#%% Compute flat norm for a region

for k,region in enumerate(region_list):

    # get the actual network edges in the region
    reg_actgeom = [g for g in actgeom if g.intersects(region)]
    reg_syngeom = [g for g in syngeom if g.intersects(region)]
    sorted_act_geom = sort_geometry(reg_actgeom)
    sorted_syn_geom = sort_geometry(reg_syngeom)
    print("Task completed: Actual and synthetic network geometries sorted")
    
    
    # Flat norm computation
    D = perform_triangulation(sorted_act_geom,sorted_syn_geom,adj=1000)
    if D['triangulated'] == None:
        fig_ = plot_failed_triangulation(D)
        sys.exit(0)
    
    
    T1 = get_current(D['triangulated'], D['actual'])
    T2 = get_current(D['triangulated'], D['synthetic'])
    T = T1 - T2
    print("Task completed: obtained current information")
    
    
    # Perform flat norm computation
    fig = plt.figure(figsize=(60,20))
    # Plot 2: All segments and points in the pre-triangulated phase
    ax1 = fig.add_subplot(2,6,1)
    ax1 = plot_intermediate_result(D["intermediate"],ax1)
    # Plot 3: Post-triangulation phase with currents
    ax2 = fig.add_subplot(2,6,7)
    ax2 = plot_triangulation(D["triangulated"],T1,T2,ax2)
    for i,lambda_ in enumerate(np.linspace(1000,100000,10)):
        x,s,norm = msfn(D['triangulated']['vertices'], D['triangulated']['triangles'], 
                        D['triangulated']['edges'], T, lambda_,k=np.pi/(180.0))
        
        print("The computed simplicial flat norm is:",norm)
        # Show results
        quo = int(i / 5)
        rem = i % 5
        ax = fig.add_subplot(2,6,(6*quo)+(rem+2))
        plot_norm(D["triangulated"],x,s,ax)
        ax.set_title("Flat norm scale, lambda = "+str(int(lambda_)), fontsize=20)
        
    
    filename = area+'-region-norm-'+str(k+1)
    fig.savefig(workpath+'/figs/'+filename+'.png',bbox_inches='tight')


