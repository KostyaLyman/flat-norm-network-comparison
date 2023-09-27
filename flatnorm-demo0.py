# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:23:40 2022

Author: Rounak Meyur

Description: Demonstrates the simplicial flat norm for a given current.
"""

from __future__ import absolute_import

import sys,os
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString
import triangle as tr
import matplotlib.pyplot as plt

workpath = os.getcwd()


from libs.pyDrawNetworklib import plot_demo_flatnorm
from libs.pyFlatNormlib import get_current, msfn, get_structure


#%% Flat norm theory - Get the input geometry
import scipy.stats as stats
import math

mu1 = 0
variance1 = 49
sigma1 = math.sqrt(variance1)
mu2 = 100
variance2 = 225
sigma2 = math.sqrt(variance2)

xpts = np.linspace(mu1 - 10*sigma1, mu2 + 7*sigma2, 200)
ypts = 1000*(stats.norm.pdf(xpts, mu1, sigma1)+stats.norm.pdf(xpts, mu2, sigma2))


# Get input geometry
vertices = [Point(x,y) for x,y in zip(xpts,ypts)]
line = LineString(vertices)
geom = [LineString((pt1,pt2)) for pt1,pt2 in zip(vertices,vertices[1:])]


# Structure with added segments
all_lines = MultiLineString(geom)
rect_env = list(all_lines.minimum_rotated_rectangle.buffer(1e-4).boundary.coords)
extra_geom = [LineString((pt1,pt2)) for pt1,pt2 in zip(rect_env,rect_env[1:])]
struct = get_structure(extra_geom + geom)

# Triangulated structure
tri_struct = tr.triangulate(struct,opts='ps')
edges = []
for tgl in tri_struct['triangles']:
    if ([tgl[0],tgl[1]] not in edges) and ([tgl[1],tgl[0]] not in edges):
        edges.append([tgl[0],tgl[1]])
    if ([tgl[1],tgl[2]] not in edges) and ([tgl[2],tgl[1]] not in edges):
        edges.append([tgl[1],tgl[2]])
    if ([tgl[2],tgl[0]] not in edges) and ([tgl[0],tgl[2]] not in edges):
        edges.append([tgl[2],tgl[0]])
tri_struct['edges'] = np.array(edges)


# Input current
T = get_current(tri_struct, geom)

#%% Flatnorm computation
lambda_ = 0.03
x,s,norm,_,_,_ = msfn(tri_struct['vertices'], tri_struct['triangles'], 
                tri_struct['edges'], T, lambda_,k=1)


#%% Plot the results
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(211)
ax = plot_demo_flatnorm(line,tri_struct,x,s,ax,offset=2)
filename = 'demo-0'
fig.savefig(workpath+'/figs/demo/'+filename+'.png',bbox_inches='tight')