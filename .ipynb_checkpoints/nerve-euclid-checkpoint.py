# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:02:08 2022

Author: Rounak Meyur
"""

import sys,os
import numpy as np
import networkx as nx
from shapely.geometry import LineString,Point,MultiLineString, LinearRing
from itertools import product
import geopandas as gpd
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic

workpath = os.getcwd()
figpath = workpath + "/figs/"
tmppath = workpath + "/temp/"


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
    d = {'nodes':range(len(points)),
         'geometry':[pt_geom for pt_geom in points]}
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
    if len(lines) == 0:
        return ax
    d = {'edges':range(len(lines)),
         'geometry':[line_geom for line_geom in lines]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha)
    return ax

def get_arcs(seed,radius,geom):
    # Get number of points to be interpolated
    n_pts = 2*len(geom) + 1
    
    # Construct the linear ring and interpolate points
    lr = LinearRing(seed.buffer(radius).exterior.coords)
    pts = [lr.interpolate((i/n_pts),normalized=True) for i in range(n_pts+1)]
    
    # Construct the edgelist
    arc_geom = [LineString([pts[k],pts[k+1]]) for k in range(n_pts)]
    return arc_geom


def line_intersection(line1, line2):
    xdiff = (line1.coords[0][0] - line1.coords[1][0], 
             line2.coords[0][0] - line2.coords[1][0])
    ydiff = (line1.coords[0][1] - line1.coords[1][1], 
             line2.coords[0][1] - line2.coords[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1.coords), det(*line2.coords))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return Point(x, y)

def Line2Vector(line,pt):
    p = np.array(pt.coords[0])
    l = np.array(line.xy)
    if pt.distance(Point(line.coords[0])) < 1e-6:
        v = l[:,1] - p
    else:
        v = l[:,0] - p
    return v/np.linalg.norm(v)
    
    

def bisector(line1,line2,pt_int):
    # Vector representing point of intersection
    p = np.array(pt_int.coords[0])
    # Vectors representing lines
    a = Line2Vector(line1,pt_int)
    b = Line2Vector(line2,pt_int)
    # Compute the bisector geometry
    c = a + b + p
    return LineString([pt_int,Point(c)])


def end_perpendicular(line):
    a = np.array(line.coords[1]) - np.array(line.coords[0])
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    c1 = b/np.linalg.norm(b)+np.array(line.coords[1])
    c0 = b/np.linalg.norm(b)+np.array(line.coords[0])
    return [LineString([Point(line.coords[0]),Point(c0)]),
            LineString([Point(line.coords[1]),Point(c1)])]


def voronoi(line1, line2):
    try:
        # Intersecting point of the edges
        p_intersection = line_intersection(line1, line2)
        b_geom = bisector(line1,line2,p_intersection)
    except:
        # Exception for parallel lines/ collinear lines
        if check_parallel(line1, line2):
            b_geom = bisector_parallel(line1,line2)
        else:
            return line1.intersection(line2).boundary
        
    # Get the intersection points of angle bisector and perpendiculars to edges
    epsilon = 1e-4
    seg1 = LineString([line_intersection(b_geom, p_geom) \
                       for p_geom in end_perpendicular(line1)]).buffer(epsilon)
    seg2 = LineString([line_intersection(b_geom, p_geom) \
                       for p_geom in end_perpendicular(line2)]).buffer(epsilon)
    
    return seg1.intersection(seg2).boundary


def check_parallel(line1,line2):
    xdiff = (line1.coords[0][0] - line1.coords[1][0], 
             line1.coords[1][0] - line2.coords[0][0])
    ydiff = (line1.coords[0][1] - line1.coords[1][1], 
             line1.coords[1][1] - line2.coords[0][1])
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    div = det(xdiff, ydiff)
    if div == 0:
        return False
    else:
        return True
        
def bisector_parallel(line1,line2):
    xmean = (0.5*(line1.coords[0][0] + line2.coords[0][0]), 
             0.5*(line1.coords[1][0] + line2.coords[1][0]))
    ymean = (0.5*(line1.coords[0][1] + line2.coords[0][1]), 
             0.5*(line1.coords[1][1] + line2.coords[1][1]))
    return LineString([Point(xmean[0],ymean[0]),Point(xmean[1],ymean[1])])


#%% Skeletons
def one_skeleton(geom):
    return {(i,j):geom[i].distance(geom[j]) for i in range(len(geom)) \
            for j in range(len(geom))}


def two_skeleton(geom):
    epsilon = 1e-4
    N = len(geom)
    T = {}
    for i in range(N):
        for j in range(N):
            if i==j:
                b_ij = geom[i]
            else:
                b_ij = voronoi(geom[i],geom[j])
            for k in range(N):
                if j==k:
                    b_jk = geom[j]
                else:
                    b_jk = voronoi(geom[j],geom[k])
                if isinstance(b_jk,MultiLineString) or isinstance(b_ij,MultiLineString):
                    T[(i,j,k)] = float('inf')
                else:
                    b_ijk = b_ij.buffer(epsilon).intersection(b_jk.buffer(epsilon))
                    T[(i,j,k)] = (b_ijk.distance(geom[i]),b_ijk.distance(geom[j]),
                              b_ijk.distance(geom[k]))
    return T

#%% Bisector of two edges
# edge1 = LineString([Point(1.125,-0.125),Point(2,-0.125)])
# edge2 = LineString([Point(1.125,-0.125),Point(1.125,1.025)])

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1,1,1)

# d = {'lines':range(2),
#       'geometry':[edge1,edge2]}
# df_lines = gpd.GeoDataFrame(d, crs="EPSG:4326")
# df_lines.plot(ax=ax,color='red')

# bis = voronoi(edge1, edge2)

# if isinstance(bis,LineString):
#     d = {'lines':range(1),
#           'geometry':[bis]}
#     df_lines = gpd.GeoDataFrame(d, crs="EPSG:4326")
#     df_lines.plot(ax=ax,color='green')

# sys.exit(0)

#%% Test graphs
# Graph A
edges_A = [(1,2),(2,9),(9,10),(10,3),(3,4),(2,5),(2,6),(3,7),(3,8)]
cords_A = {1:(-2,0.125), 2:(-1.125,0.125), 5:(-1.125,1.025), 6:(-1.125,-1),
           3:(1.125,-0.125), 4:(2,-0.125), 7:(1.125,1.025), 8:(1.125,-1),
           9:(0,0.125), 10:(0,-0.125)}
edge_geom_A = {e:LineString([cords_A[e[0]],cords_A[e[1]]]) for e in edges_A}

A = nx.Graph()
A.add_edges_from(edges_A)
nx.set_edge_attributes(A,edge_geom_A,'geometry')
nx.set_node_attributes(A,cords_A,'cord')
geom_A = [A.edges[e]['geometry'] for e in A.edges]

# Graph B
edges_B = [(11,12),(12,13),(13,14),(12,15),(12,16),(13,17),(13,18)]
cords_B = {11:(-2,0), 12:(-1,0), 15:(-1,1), 16:(-1,-1),
           13:(1,0), 14:(2,0), 17:(1,1), 18:(1,-1)}
edge_geom_B = {e:LineString([cords_B[e[0]],cords_B[e[1]]]) for e in edges_B}

B = nx.Graph()
B.add_edges_from(edges_B)
nx.set_edge_attributes(B,edge_geom_B,'geometry')
nx.set_node_attributes(B,cords_B,'cord')
geom_B = [B.edges[e]['geometry'] for e in B.edges]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)

draw_edges(ax,A,color='red',width=2.0,style='solid')
draw_edges(ax,B,color='blue',width=2.0,style='solid')
draw_nodes(ax,A,color='red',size=30,marker='s',alpha=0.6)
draw_nodes(ax,B,color='blue',size=30,marker='o')


#%% Filtration: Disc of radius r at a point 

s = Point(0,0)
arc = get_arcs(s,0.5,geom_A)

draw_lines(ax,arc,color='green',alpha=0.8)


D = one_skeleton(geom_A+arc)
# T = two_skeleton(geom_A+arc)

geom = geom_A+arc
exceptions = []
epsilon = 1e-4
N = len(geom)
T = {}
for i in range(N):
    for j in range(N):
        if i==j:
            b_ij = geom[i]
        else:
            b_ij = voronoi(geom[i],geom[j])
        for k in range(N):
            if j==k:
                b_jk = geom[j]
            else:
                b_jk = voronoi(geom[j],geom[k])
            try:
                b_ijk = b_ij.buffer(epsilon).intersection(b_jk.buffer(epsilon)).boundary
                T[(i,j,k)] = (b_ijk.distance(geom[i]),b_ijk.distance(geom[j]),
                          b_ijk.distance(geom[k]))
            except:
                seg = [[geom[i],geom[j],geom[k],b_ij,b_jk]]
                exceptions.append(seg)
                T[(i,j,k)] = float('inf')












