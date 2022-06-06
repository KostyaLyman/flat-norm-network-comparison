# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:16:52 2021

Author: Rounak
Description: Includes methods to compare pair of networks on the basis of 
partitioning into multiple grids and comparing each grid. Includes methods to
compare actual and synthetic networks of Blacksburg, VA.
"""

import os,sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
actpath = workpath + "/input/actual/"
synpath = workpath + "/input/primnet/"

sys.path.append(libpath)
from pyExtractDatalib import GetDistNet,get_areadata
from pyGeometrylib import partitions
from pyDrawNetworklib import plot_deviation
from pyComparisonlib import compute_hausdorff
print("Imported modules")


#%% Data Extraction
sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]
synth_net = GetDistNet(synpath,sublist)
print("Synthetic network extracted")

#areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
areas = {'patrick_henry':194,'mcbryde':9001}

area_data = {area:get_areadata(actpath,area,root,synth_net) \
                      for area,root in areas.items()}
print("Area Data extracted and stored")

c_actual = 'orangered'
c_synth = 'blue'

sys.exit(0)


#%% Create graph
def create_graph(data):
    act_edges = data['df_lines']
    prefix = data['root']
    count = 50000
    
    for i in range(len(act_edges)):
        edge_id = act_edges["ID"][i]
        edge_geom = act_edges["geometry"][i]

#%% Functions for comparing actual and synthetic networks
# Exract area data
def get_limits(area_data):
    # Get limits for the geographical region
    lims = np.empty(shape=(len(area_data),4))
    for i,area in enumerate(area_data):
        lims[i,:] = np.array(area_data[area]['limits'])
    LEFT = np.min(lims[:,0]); RIGHT = np.max(lims[:,1])
    BOTTOM = np.min(lims[:,2]); TOP = np.max(lims[:,3])
    return LEFT,RIGHT,BOTTOM,TOP

def get_edges(area_data):
    act_edges = [g for area in area_data \
                 for g in area_data[area]['df_lines']['geometry'].tolist()]
    syn_edges = [g for area in area_data \
                 for g in area_data[area]['df_synth']['geometry'].tolist()]
    return act_edges,syn_edges

def get_nodes(area_data):
    act_nodes_geom = [g for area in area_data \
                 for g in area_data[area]['df_buses']['geometry'].tolist()]
    synth_nodes_geom = [g for area in area_data \
                 for g in area_data[area]['df_cords']['geometry'].tolist()]
    return act_nodes_geom, synth_nodes_geom

# Plot the networks
def plot_gdf(ax,df_edges,df_nodes,color):
    """"""
    df_edges.plot(ax=ax,edgecolor=color,linewidth=2.0)
    df_nodes.plot(ax=ax,color=color,markersize=500)
    return

#%% Spatial Node Distribution
def get_suffix(delta):
    if delta>0:
        suf=str(int(abs(100*delta))) + 'pos'
    elif delta<0:
        suf=str(int(abs(100*delta))) + 'neg'
    else:
        suf='00'
    return suf

leg_data = [Line2D([0], [0], color=c_actual, markerfacecolor=c_actual, 
                   marker='o',markersize=40, label='Actual distribution network'),
            Line2D([0], [0], color=c_synth, markerfacecolor=c_synth,
                   marker='o',markersize=40, label='Synthetic distribution network'),
            Patch(facecolor='white', edgecolor='black', hatch="./",
                         label='Grid cells with no actual network data')]

#%% Hausdorff distance between networks
kx = 7
ky = 7
x0 = 0.05
y0 = 0

LEFT,RIGHT,BOTTOM,TOP = get_limits(area_data)
gridlist = partitions((LEFT,RIGHT,BOTTOM,TOP),kx,ky,x0=x0,y0=y0)
act_edges,syn_edges = get_edges(area_data)
C = compute_hausdorff(gridlist,act_edges,syn_edges)
C_vals = np.array([C[bound] for bound in gridlist])
C_masked = np.ma.array(C_vals, mask=np.isnan(C_vals))

# Plot the Hausdorff deviation
fig = plt.figure(figsize=(30,25))
ax = fig.add_subplot(1,1,1)

for area in area_data:
    plot_gdf(ax,area_data[area]['df_lines'],area_data[area]['df_buses'],
                  c_actual)
    plot_gdf(ax,area_data[area]['df_synth'],area_data[area]['df_cords'],
                  c_synth)

ax.legend(handles=leg_data,ncol=1,fontsize=50, 
          bbox_to_anchor=(0.0, 0.0),loc='upper left')


plot_deviation(ax,gridlist,C_masked,colormap=cm.Greens,vmin=0,vmax=500)

cobj = cm.ScalarMappable(cmap=cm.Greens)
cobj.set_clim(vmin=0.0,vmax=500)
cbar = fig.colorbar(cobj,ax=ax)
cbar.set_label("Hausdorff distance (meters)",size=70)
cbar.ax.tick_params(labelsize=60)

suffix = "-"+get_suffix(x0)+"-"+get_suffix(y0)
fig.savefig("{}{}.png".format(figpath,'hauss-comparison-'+str(kx)+'-'+str(ky)+suffix),
        bbox_inches='tight')








