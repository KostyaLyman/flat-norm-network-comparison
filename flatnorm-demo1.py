# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:39:13 2022

Author: Rounak Meyur

Description: Shows a simple demonstration of how simplicial flat norm can be 
used to compare a pair of geometries. How the deviation between the geometries
is reflected in the computed flat norm.
"""

from __future__ import absolute_import

import sys,os
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from shapely import affinity

workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
figpath = workpath + "/figs/"

FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"


from pyFlatNormlib import get_geometry, get_current, msfn, perform_triangulation
from pyDrawNetworklib import plot_norm, plot_input

#%% Function
def compute_flatnorm(in_geom, angle, lambda_ = 0.001):
    in_geom_rot = [affinity.rotate(g, angle, (0,0)) for g in in_geom]
    # Construct the geometry list
    geom1 = []
    geom2 = []
    for g in in_geom:
        geom1.extend(get_geometry(g))
    for g in in_geom_rot:
        geom2.extend(get_geometry(g))
    
    # Compute flat norm
    D = perform_triangulation(geom1,geom2,adj=1)
    T1 = get_current(D['triangulated'], D['actual'])
    T2 = get_current(D['triangulated'], D['synthetic'])
    x,s,norm,_,_,_ = msfn(D['triangulated']['vertices'], 
                        D['triangulated']['triangles'], 
                        D['triangulated']['edges'], 
                        T1-T2, lambda_,k=1)
    
    # Plot the result
    fig = plt.figure(figsize=(13,6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    ax1 = plot_input(geom1,geom2,ax1)
    ax1.set_xlim(-1.1,1.1)
    ax1.set_ylim(-1.1,1.1)
    ax2 = plot_norm(D["triangulated"],x,s,ax2)
    ax2.set_xlim(-1.1,1.1)
    ax2.set_ylim(-1.1,1.1)
    
    title1 = f"Angle= {angle : 0.1f} degrees"
    title2 = f"Flat norm ${FN}$= {norm : 0.2f}"
    suptitle = f"{title1} : {title2}"
    fig.suptitle(suptitle, fontsize=33)
    return fig

#%% Main code

anglist = [90,60,30,15]

filename = 'testnorm'

in_geom1 = [LineString((Point(-1,0),Point(1,0)))]
for i,ang in enumerate(anglist):
    fig = compute_flatnorm(in_geom1, ang)
    fig.savefig(f"{figpath}{filename}-{i+1}.png",bbox_inches='tight')
    




