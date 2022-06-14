#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:35:52 2022

@author: Rounak Meyur
Description: Using voronoi partitions to create bisectors
"""

import sys,os
import numpy as np
import networkx as nx
from shapely.geometry import LineString,Point,MultiLineString,LinearRing
from itertools import product
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
import geopandas as gpd
from scipy.spatial import Voronoi

workpath = os.getcwd()
figpath = workpath + "/figs/"
tmppath = workpath + "/temp/"


#%% Functions to convert lines to points
def interpolate(line_geom,num_pts=100,ref='A'):
    points = {}
    dist = 1/num_pts
    for f in range(num_pts+1):
        x,y = line_geom.interpolate(f*dist,normalized=True).xy
        xy = (x[0],y[0])
        points[ref+str(f)] = Point(xy)
    return points

def intersect(geom_a,geom_b,eps):
    return geom_a.buffer(eps).intersects(geom_b.buffer(eps))

def intersection(geom_a,geom_b,eps=1e-4):
    return geom_a.buffer(eps).intersection(geom_b.buffer(eps)).boundary

def get_arcs(seed,radius,geom):
    # Get number of points to be interpolated
    n_pts = 2*len(geom) + 1
    
    # Construct the linear ring and interpolate points
    lr = LinearRing(seed.buffer(radius).exterior.coords)
    pts = [lr.interpolate((i/n_pts),normalized=True) for i in range(n_pts+1)]
    
    # Construct the edgelist
    arc_geom = [LineString([pts[k],pts[k+1]]) for k in range(n_pts)]
    return arc_geom

#%% Plot functions
def draw_nodes(ax,graph,nodelist=None,color='red',size=30,alpha=1.0,marker='*'):
    if nodelist == None:
        nodelist = graph.nodes
    d = {'nodes':nodelist,
         'geometry':[Point(graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,marker=marker)
    return ax

def draw_points(ax,points,color='red',size=10,alpha=1.0,marker='o'):
    if len(points) == 0:
        return ax
    if isinstance(points,list):
        d = {'nodes':range(len(points)),
             'geometry':[pt_geom for pt_geom in points]}
    elif isinstance(points,dict):
        d = {'nodes':range(len(points)),
             'geometry':[points[k] for k in points]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,marker=marker)
    return ax

def draw_edges(ax,graph,edgelist=None,color='red',width=2.0,style='solid',
               alpha=1.0):
    if edgelist == []:
        return ax
    if edgelist == None:
        edgelist = graph.edges
    d = {'edges':edgelist,
         'geometry':[graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha)
    return ax

def draw_lines(ax,lines,color='red',width=2.0,style='solid',alpha=1.0):
    
    if isinstance(lines,LineString):
        lines = [lines]
    
    if len(lines) == 0:
        return ax
    
    d = {'edges':range(len(lines)),
         'geometry':[line_geom for line_geom in lines]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha)
    return ax

#%% Compute nerve homology
def voronoi_bisector(line1,line2,num_pts=20):
    pts_A = interpolate(line1,num_pts,ref='A')
    pts_B = interpolate(line2,num_pts,ref='B')
    
    pA = np.array([(pts_A[k].x,pts_A[k].y) for k in pts_A])
    pB = np.array([(pts_B[k].x,pts_B[k].y) for k in pts_B])
    points = np.concatenate((pA,pB))
    
    vor = Voronoi(points)
    lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices \
             if -1 not in line]
    
    return MultiLineString(lines)


#%% Main code
edge1 = LineString([Point(0,0),Point(1,0)])
edge2 = LineString([Point(-2,0),Point(3,0)])



fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
draw_lines(ax,[edge1,edge2],color='green',alpha=0.8)

try:
    bisector = voronoi_bisector(edge1, edge2)
except:
    print("Part of the same line segment")
    bisector = intersection(edge1,edge2)
draw_lines(ax, bisector, color='blue')



























