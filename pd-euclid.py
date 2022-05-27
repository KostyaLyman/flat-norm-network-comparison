# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:02:08 2022

Author: Rounak Meyur
"""

import sys,os
import numpy as np
import networkx as nx
from shapely.geometry import LineString,Point
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


def cech_complex(edges,pts_A,pts_B,eps):
    return [e for e in edges if intersect(pts_A[e[0]], pts_B[e[1]], eps)]


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

#%% Test geometry

A = LineString([(0,0.125),(1,0.5),(2,0.125)])
B = LineString([(0,0),(1,0.125),(2,0)])

A_pts = interpolate(A,10,ref='A')
B_pts = interpolate(B,10,ref='B')
edgeset = product(A_pts.keys(),B_pts.keys())


e_list = np.linspace(0.1,0.4,31)
for i,epsilon in enumerate(e_list):
    e_AB = cech_complex(edgeset,A_pts,B_pts,epsilon)
    plot_1simplex(A_pts,B_pts,e_AB,name=str(i),eps=epsilon)

makegif(tmppath,figpath+'1-simplices')
































# Draw the edges
# edges_A = [(list(nodes_A.keys())[i],list(nodes_A.keys())[i+1]) \
#            for i in range(len(nodes_A)-1)]
# edges_B = [(list(nodes_B.keys())[i],list(nodes_B.keys())[i+1]) \
#            for i in range(len(nodes_B)-1)]
# ax = draw_edges(ax,edges_A,all_nodes,color='blue',width=2.0,style='solid')
# ax = draw_edges(ax,edges_B,all_nodes,color='red',width=2.0,style='solid')
