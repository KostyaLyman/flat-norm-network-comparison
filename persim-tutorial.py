# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:27:09 2022

Author: Rounak Meyur

Description: Tutorial for using persistent diagram
"""

from itertools import product


import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

from ripser import ripser
from persim import plot_diagrams

import geopandas as gpd

#%% Plotting functions
def draw_nodes(ax,graph,nodelist=None,color='red',size=30,alpha=1.0,marker='*',label=None):
    if nodelist == None:
        nodelist = graph.nodes
    d = {'nodes':nodelist,
         'geometry':[Point(graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,marker=marker,label=label)
    return ax

def draw_edges(ax,graph,edgelist=None,color='red',width=2.0,style='solid',
               alpha=1.0,label=None):
    if edgelist == []:
        return ax
    if edgelist == None:
        edgelist = graph.edges
    d = {'edges':edgelist,
         'geometry':[graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha,label=label)
    return ax

def draw_points(ax,points,color='red',size=10,alpha=1.0,marker='o',label=None):
    if len(points) == 0:
        return ax
    if isinstance(points,list):
        d = {'nodes':range(len(points)),
             'geometry':[pt_geom for pt_geom in points]}
    elif isinstance(points,dict):
        d = {'nodes':range(len(points)),
             'geometry':[points[k] for k in points]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,marker=marker,label=label)
    return ax

def draw_lines(ax,lines,color='red',width=2.0,style='solid',alpha=1.0,label=None):
    if isinstance(lines,LineString):
        lines = [lines]
    if len(lines) == 0:
        return ax
    d = {'edges':range(len(lines)),
         'geometry':[line_geom for line_geom in lines]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha,label=label)
    return ax


#%% Graph example
import networkx as nx
from shapely.geometry import Point,LineString,MultiLineString

def interpolate_points(line_geom,num_pts=100):
    points = []
    dist = 1/num_pts
    for f in range(num_pts+1):
        x,y = line_geom.interpolate(f*dist,normalized=True).xy
        points.append(Point(x[0],y[0]))
    return points

# Graph A
edges_A = [(1,2),(2,3)]
cords_A = {1:(-1,0.5), 2:(0,1), 3:(1,0.5)}
edge_geom_A = {e:LineString([cords_A[e[0]],cords_A[e[1]]]) for e in edges_A}

A = nx.Graph()
A.add_edges_from(edges_A)
nx.set_edge_attributes(A,edge_geom_A,'geometry')
nx.set_node_attributes(A,cords_A,'cord')
geom_A = MultiLineString([A.edges[e]['geometry'] for e in A.edges])

# Graph B
edges_B = [(11,12),(12,13),(13,14),(14,15)]
cords_B = {11:(-1,0.25), 12:(-0.5,0.75), 13:(0,1), 14:(0.5,0.6), 15:(1,0.5)}
edge_geom_B = {e:LineString([cords_B[e[0]],cords_B[e[1]]]) for e in edges_B}

B = nx.Graph()
B.add_edges_from(edges_B)
nx.set_edge_attributes(B,edge_geom_B,'geometry')
nx.set_node_attributes(B,cords_B,'cord')
geom_B = MultiLineString([B.edges[e]['geometry'] for e in B.edges])



#%% Interpolate points
pts_A = interpolate_points(geom_A,num_pts=5)
pts_B = interpolate_points(geom_B,num_pts=5)    

fig = plt.figure(figsize=(30,30))
ax1 = fig.add_subplot(1,2,1)

draw_edges(ax1,A,color='red',width=2.0,style='solid',label="Graph A edges")
draw_edges(ax1,B,color='blue',width=2.0,style='solid',label="Graph B edges")
draw_nodes(ax1,A,color='red',size=30,marker='s',alpha=0.6,label="Graph A nodes")
draw_nodes(ax1,B,color='blue',size=30,marker='o',label="Graph B nodes")
ax1.legend(markerscale=2,fontsize=20)

ax2 = fig.add_subplot(1,2,2)
draw_points(ax2,pts_A,color='red',size=25,alpha=0.6,marker='D',label="Point Cloud A")
draw_points(ax2,pts_B,color='blue',size=25,alpha=0.6,marker='o',label="Point Cloud B")
ax2.legend(markerscale=3,fontsize=25)


#%% RIPSER homology persistence diagram
lA = [list(pt.coords)[0] for pt in pts_A]
lB = [list(pt.coords)[0] for pt in pts_B]
# l = lA + lB

dataA = np.array(lA)
dataB = np.array(lB)

dgmsA = ripser(dataA, thresh=2.0, maxdim=1)['dgms']
dgmsB = ripser(dataB, thresh=2.0, maxdim=1)['dgms']


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
plot_diagrams(dgmsA,ax=ax1,size=100)
plot_diagrams(dgmsB,ax=ax2,size=100)


#%% Small Example
l = [[0,0],[0,3],[4,0],[4,3]]
# l = [[0,0],[0,3],[4,0]]
data = np.array(l)

dgms = ripser(data, coeff=2, thresh=10.0, maxdim=1)['dgms']

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plot_diagrams(dgms,ax=ax,size=100)
ax.legend(fontsize=25,markerscale=2)












