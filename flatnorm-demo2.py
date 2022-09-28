# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:55:10 2022

Author: Rounak Meyur

Description: Shows a simple demonstration of simplicial flat norm computation 
for a pair of geometries.
"""

from __future__ import absolute_import

import sys,os
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt


workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
figpath = workpath + "/figs/"


from pyFlatNormlib import get_geometry, get_current, msfn, perform_triangulation
from pyDrawNetworklib import plot_norm, plot_input, plot_failed_triangulation
from pyDrawNetworklib import plot_intermediate_result, plot_triangulation


# Input geometries
in_geom1 = [LineString((Point(0,0),Point(1,1))),
         LineString((Point(1,1),Point(2,0))),
         LineString((Point(2,0),Point(3,1),Point(4,0)))]
in_geom2 = [LineString((Point(0,0.5),Point(2,0.5),Point(4,0.5)))]

geom1 = []
geom2 = []
for g in in_geom1:
    geom1.extend(get_geometry(g))
for g in in_geom2:
    geom2.extend(get_geometry(g))

D = perform_triangulation(geom1,geom2,adj=1)


if D['triangulated'] == None:
    plot_failed_triangulation(D)
    sys.exit(0)


T1 = get_current(D['triangulated'], D['actual'])
T2 = get_current(D['triangulated'], D['synthetic'])

print("Task completed: obtained current information")

T = T1 - T2

print(sum(abs(T1)))
print(len(D['actual']))
print(sum(abs(T2)))
print(len(D['synthetic']))

lambda_ = 0.001
x,s,norm,_,_ = msfn(D['triangulated']['vertices'], D['triangulated']['triangles'], 
                D['triangulated']['edges'], T, lambda_,k=1)

print("The computed simplicial flat norm is:",norm)

#%% Plot the example  
fig = plt.figure(figsize=(20,10))
# Plot 1: Plot the geometries of the pair of networks
ax1 = fig.add_subplot(221)
ax1 = plot_input(geom1,geom2,ax1)
# Plot 2: All segments and points in the pre-triangulated phase
ax2 = fig.add_subplot(222)
ax2 = plot_intermediate_result(D["intermediate"],ax2)
# Plot 3: Post-triangulation phase with currents
ax3 = fig.add_subplot(223)
ax3 = plot_triangulation(D["triangulated"],T1,T2,ax3)
# Plot 4: flat-norm computation
ax4 = fig.add_subplot(224)
ax4 = plot_norm(D["triangulated"],x,s,ax4)
filename = 'demo-1'
fig.savefig(workpath+'/figs/'+filename+'.png',bbox_inches='tight')