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


def plot_1simplex(nodes_A,nodes_B,edges_AB,eps=None,name='tmp-0',close=True):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1,1,1)
    
    # Draw the epsilon radius
    if eps!=None:
        pts = {'A':Point(1,0.5),'B':Point(1,0.125)}
        ax = plot_buffer(ax,pts,eps)
    
    # Draw the interpolated points
    draw_nodes(ax,nodes_A,color='blue',size=30,marker='o')
    draw_nodes(ax,nodes_B,color='red',size=30,marker='*')
    
    # Get all node coordinates
    all_nodes = {p:nodes_A[p] for p in nodes_A}
    for p in nodes_B: 
        all_nodes[p]=nodes_B[p]
    
    # Draw the simplices
    ax = draw_edges(ax,edges_AB,all_nodes,color='green',width=2.0,style='solid')
    
    # Draw the edges
    edges_A = [(list(nodes_A.keys())[i],list(nodes_A.keys())[i+1]) \
                for i in range(len(nodes_A)-1)]
    edges_B = [(list(nodes_B.keys())[i],list(nodes_B.keys())[i+1]) \
                for i in range(len(nodes_B)-1)]
    ax = draw_edges(ax,edges_A,all_nodes,color='blue',width=2.0,style='solid')
    ax = draw_edges(ax,edges_B,all_nodes,color='red',width=2.0,style='solid')
    
    # Common axes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(-0.8,1.2)
    
    # Save the image in temp directory
    fig.savefig("{}{}.png".format(tmppath,name),bbox_inches='tight')
    if close: plt.close()
    return


def plot_buffer(ax,pts,eps):
    d = {'nodes':[n for n in pts],
         'geometry':[pts[n].buffer(eps) for n in pts]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color='orange',alpha=0.1)
    return ax


def bread_crumbs(s,geom,r,d):
    crumbs = []
    k = 0
    while (k*d < r):
        # Get the linear ring of the point buffer
        pt = s.buffer(k*d)
        lr = LinearRing(pt.exterior.coords)
        
        # Get intersections
        pts = lr.intersection(geom)
        
        # Add bread crumbs if intersections present
        if not pts.is_empty:
            for p in pts:
                crumbs.append(p)
        
        # Increment the integer k
        k += 1
    return crumbs

def get_distance(pts_A,pts_B,delta):
    dist_A = [min([pt_a.distance(pt_b) for pt_b in pts_B]) for pt_a in pts_A]
    dist_B = [min([pt_b.distance(pt_a) for pt_a in pts_A]) for pt_b in pts_B]
    
    kA = len([d for d in dist_A if d<=delta])
    kB = len([d for d in dist_B if d<=delta])
    
    prec = kA/len(pts_A)
    rec = kB/len(pts_B)
    return prec,rec

#%% Plot the persistence diagram
import imageio

def makegif(src,dest):
    '''
    Input:  src : Source directory of images
            dest: Destination path of gif
    '''
    fnames = [f for f in os.listdir(src) if ".png" in f]
    fnames_sorted = [str(m)+'.png'for m in 
                     sorted([int(s.strip('.png')) for s in fnames])]
    

    with imageio.get_writer(dest+'.gif', mode='I') as writer:
        for f in fnames_sorted:
            image = imageio.imread(src+f)
            writer.append_data(image)
    
    for f in fnames:
        os.remove(src+f)
    return



#%% Test graphs
# Graph A
edges_A = [(1,2),(2,3),(3,4),(2,5),(2,6),(3,7),(3,8)]
cords_A = {1:(-2,0.125), 2:(-1.125,0.125), 5:(-1.125,1.025), 6:(-1.125,-1),
           3:(1.125,-0.125), 4:(2,-0.125), 7:(1.125,1.025), 8:(1.125,-1)}
edge_geom_A = {}
for e in edges_A:
    if e == (2,3):
        edge_geom_A[e] = LineString([cords_A[2],(0,0.125),
                                     (0,-0.125),cords_A[3]])
    else:
        edge_geom_A[e] = LineString([cords_A[e[0]],cords_A[e[1]]])

A = nx.Graph()
A.add_edges_from(edges_A)
nx.set_edge_attributes(A,edge_geom_A,'geometry')
nx.set_node_attributes(A,cords_A,'cord')
geom_A = MultiLineString([A.edges[e]['geometry'] for e in A.edges])

# Graph B
edges_B = [(11,12),(12,13),(13,14),(12,15),(12,16),(13,17),(13,18)]
cords_B = {11:(-2,0), 12:(-1,0), 15:(-1,1), 16:(-1,-1),
           13:(1,0), 14:(2,0), 17:(1,1), 18:(1,-1)}
edge_geom_B = {e:LineString([cords_B[e[0]],cords_B[e[1]]]) for e in edges_B}

B = nx.Graph()
B.add_edges_from(edges_B)
nx.set_edge_attributes(B,edge_geom_B,'geometry')
nx.set_node_attributes(B,cords_B,'cord')
geom_B = MultiLineString([B.edges[e]['geometry'] for e in B.edges])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)

draw_edges(ax,A,color='red',width=2.0,style='solid')
draw_edges(ax,B,color='blue',width=2.0,style='solid')
draw_nodes(ax,A,color='red',size=30,marker='s',alpha=0.6)
draw_nodes(ax,B,color='blue',size=30,marker='o')

# Test bread crumb code
seed = Point(1,0)
crumbs_A = bread_crumbs(seed, geom_A, 0.5, 0.1)
crumbs_B = bread_crumbs(seed, geom_B, 0.5, 0.1)


draw_points(ax,crumbs_A,color='red',size=20,marker='s',alpha=0.5)
draw_points(ax,crumbs_B,color='blue',size=20,marker='o',alpha=0.5)



get_distance(crumbs_A, crumbs_B, 0.12)

























