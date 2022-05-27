# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:02:08 2022

Author: Rounak Meyur
"""

import sys,os
import numpy as np
import networkx as nx
from shapely.geometry import LineString,Point
import geopandas as gpd
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic


#%% Classes
def interpolate(line_geom,num_pts=100,ref='A'):
    points = {}
    dist = 1/num_pts
    for f in range(num_pts):
        x,y = line_geom.interpolate(f*dist,normalized=True).xy
        xy = (x[0],y[0])
        points[ref+str(f)] = Point(xy)
    return points

def intersect(pt_a,pt_b,eps):
    return pt_a.buffer(eps).intersects(pt_b.buffer(eps))


def cech_complex(pts_A,pts_B,eps):
    edges_AB = [(a,b) for a in pts_A for b in pts_B \
                     if intersect(pts_A[a], pts_B[b], eps)]
    return edges_AB

def draw_nodes(ax,dict_nodes,color='red',size=30,alpha=1.0,marker='*'):
    d = {'nodes':[n for n in dict_nodes],
         'geometry':[dict_nodes[n] for n in dict_nodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,marker=marker)
    return ax

def draw_edges(ax,edgelist,dict_nodes,color='red',width=2.0,style='solid',
               alpha=1.0):
    if edgelist == []:
        return ax
    d = {'edges':edgelist,
         'geometry':[LineString((dict_nodes[e[0]],dict_nodes[e[1]])) \
                     for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha)
    return ax


def plot_cech(nodes_A,nodes_B,edges_AB):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1,1,1)
    # Draw the interpolated points
    draw_nodes(ax,nodes_A,color='blue',size=30,marker='o')
    draw_nodes(ax,nodes_B,color='red',size=30,marker='*')
    
    # Get all node coordinates
    all_nodes = {p:nodes_A[p] for p in nodes_A}
    for p in nodes_B: 
        all_nodes[p]=nodes_B[p]
    
    # Draw the edges
    edges_A = [(list(nodes_A.keys())[i],list(nodes_A.keys())[i+1]) \
               for i in range(len(nodes_A)-1)]
    edges_B = [(list(nodes_B.keys())[i],list(nodes_B.keys())[i+1]) \
               for i in range(len(nodes_B)-1)]
    ax = draw_edges(ax,edges_A,all_nodes,color='blue',width=2.0,style='solid')
    ax = draw_edges(ax,edges_B,all_nodes,color='red',width=2.0,style='solid')
    ax = draw_edges(ax,edges_AB,all_nodes,color='green',width=2.0,style='solid')
    return ax
    

#%% Test geometry

A = LineString([(0,0.125),(1,0.5),(2,0.125)])
B = LineString([(0,0),(1,0.125),(2,0)])

A_pts = interpolate(A,20,ref='A')
B_pts = interpolate(B,20,ref='B')



for epsilon in [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]:
    e_AB = cech_complex(A_pts,B_pts,epsilon)
    
    ax = plot_cech(A_pts,B_pts,e_AB)



#%% Plot the persistence diagram
































