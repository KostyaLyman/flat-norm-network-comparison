# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:23:40 2022

Author: Rounak Meyur
"""

from __future__ import absolute_import

import sys,os
import numpy as np
from scipy import sparse
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import triangle as tr


workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
actpath = workpath + "/input/actual/"
synpath = workpath + "/input/primnet/"

from pyUtilslib import simpvol, boundary_matrix
from pyLPsolverlib import lp_solver


#%% Plot functions

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

def draw_lines(ax,lines,color='red',width=2.0,style='solid',alpha=1.0,label=None,
               directed=False):
    if isinstance(lines,LineString):
        lines = [lines]
    if len(lines) == 0:
        return ax
    d = {'edges':range(len(lines)),
         'geometry':[line_geom for line_geom in lines]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha,label=label)
    if directed:
        for line_geom in lines:
            arrow_width=0.03
            head_width = 2.5 * arrow_width
            head_length = 2 * arrow_width
            
            cp0 = np.array(line_geom.coords[0])
            cp1 = np.mean(line_geom.coords,axis=0)
            
            delta = cos, sin = (cp1 - cp0) / np.hypot(*(cp1 - cp0))
            length = Point(cp0).distance(Point(cp1))
            x_pos, y_pos = (
            (cp0 + cp1) / 2  - delta * length / 2)
            ax.arrow(x_pos, y_pos, cos * length, sin * length,ec=color,fc=color,
                     head_width=head_width, head_length=head_length,ls='--', 
                     shape='full',length_includes_head=True)
    return ax



#%% MSFN definition
def msfn(points, simplices, subsimplices, input_current, lambda_, w=[], v=[], cons=[]):
    '''
    MSFN - Multiscale flat norm
    Accepts simplicial settings, an input current, multiscale factor(:math:`\lambda`).
    Returns flat norm decomposition of the input current and the flat norm.
    Let K be an underlying simplicial complex of dimension q.
    :param float points: points in K.
    :param int simplices: (p+1)-simplices in K, an array of dimension (nx(p+1)) 
        where :math:`p+1 \leq q` and n is the number of p+1-simplices in K. 
    :param int subsimplices: p-simplices in K, an array of dimension (mxp) 
        where :math:`p \leq q` and m is the number of p-simplices in K. 
    :param int input_current: a vector for an input current. 
    :param float lambda_: multiscale factor.  
    :param float w: a vector of subsimplices volumes.
    :param float v: a vector of simplices volumes.
    :param int cons: a constraint matrix A.
    :returns: x, s, norm-- p-chain, (p+1)-chain of flat norm decompostion, flat norm.
    :rtype: int, int, float.
    '''
    m_subsimplices = subsimplices.shape[0]
    n_simplices = simplices.shape[0]
    if w == []:
        w = simpvol(points, subsimplices)
    if v == []:
        v = simpvol(points, simplices)
    if cons == []:
        b_matrix = boundary_matrix(simplices, subsimplices, format='coo')
        m_subsimplices_identity = sparse.identity(m_subsimplices, dtype=np.int8, format='coo')
        cons = sparse.hstack((m_subsimplices_identity, -m_subsimplices_identity, b_matrix, -b_matrix))
    c = np.concatenate((abs(w), abs(w), lambda_*abs(v), lambda_*abs(v))) 
    c.reshape(len(c),1)
    sol, norm = lp_solver(c, cons, input_current)
    x = (sol[0:m_subsimplices] - sol[m_subsimplices:2*m_subsimplices]).reshape((1,m_subsimplices)).astype(int)
    s = (sol[2*m_subsimplices:2*m_subsimplices+n_simplices] - sol[2*m_subsimplices+n_simplices:]).reshape(1, n_simplices).astype(int)
    return x, s, norm

def get_flat_approx(x,points,simplices):
    flat_lines = []
    m = simplices.shape[0]
    for i in range(m):
        if x[0,i] == 1:
            flat_lines.append(LineString(points[simplices[i,:]]))
        elif x[0,i] == -1:
            flat_lines.append(LineString(points[simplices[i,:]][::-1]))
    return flat_lines

#%% Extract actual network data and perform constrained Delaunay triangulation
areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
area = 'mcbryde'

# Create actual graph network
df_lines = gpd.read_file(actpath+area+'/'+area+'_edges.shp')
all_geom = [df_lines['geometry'][i] for i in range(len(df_lines))]

vertices = []
segments = []
for geom in all_geom:
    geom_vertices = [Point(c) for c in geom.coords]
    for c in geom_vertices:
        if c not in vertices:
            vertices.append(c)
    for i in range(len(geom_vertices)-1):
        ind1 = vertices.index(geom_vertices[i])
        ind2 = vertices.index(geom_vertices[i+1])
        segments.append((ind1,ind2))

# Constrained Delaunay Triangulation
actual = {'vertices':np.array([v.coords[0] for v in vertices]), 
          'segments':np.array(segments)}
t = tr.triangulate(actual,'pc')
geom_segments = [LineString((vertices[c[0]],vertices[c[1]])) for c in segments]

#%% Get actual set of vertices, simplices and subsimplices after the triangulation
vertices = t['vertices']
simplices = t['triangles']
subsimplices = []
for tgl in t['triangles']:
    if ([tgl[0],tgl[1]] not in subsimplices) and ([tgl[1],tgl[0]] not in subsimplices):
        subsimplices.append([tgl[0],tgl[1]])
    if ([tgl[1],tgl[2]] not in subsimplices) and ([tgl[2],tgl[1]] not in subsimplices):
        subsimplices.append([tgl[1],tgl[2]])
    if ([tgl[2],tgl[0]] not in subsimplices) and ([tgl[0],tgl[2]] not in subsimplices):
        subsimplices.append([tgl[2],tgl[0]])
subsimplices = np.array(subsimplices)

# Update the geometries of post triangulation vertices and subsimplices
geom_vertices = [Point(v) for v in vertices]
geom_subsimplices = [LineString((vertices[c[0]],vertices[c[1]])) for c in subsimplices]

# Update the segment list
org_segments = [[Point(actual['vertices'][c]) for c in seg] \
                for seg in actual['segments'].tolist()]

segments = []
for geom_seg in org_segments:
    ind1 = geom_vertices.index(geom_seg[0])
    ind2 = geom_vertices.index(geom_seg[1])
    segments.append([ind1,ind2])

#%% Plot the geometries
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(211)
draw_points(ax1,geom_vertices,color='black',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom_subsimplices,color='black',width=1.0,style='dashed',alpha=1.0,
           directed=False)
ax1.set_title("Simplicial complexes",fontsize=45)
ax1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax2 = fig.add_subplot(212)
draw_points(ax2,geom_vertices,color='blue',size=20,alpha=1.0,marker='o')
draw_lines(ax2,geom_subsimplices,color='black',width=1.0,style='dashed',alpha=0.5,
           directed=False)
draw_lines(ax2,geom_segments,color='blue',width=1.0,style='solid',alpha=1.0,directed=False)
ax2.set_title("Input current",fontsize=45)
ax2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

#%% Input geometry
points = t['vertices']

# input current
input_current = []
m = subsimplices.shape[0]
for i in range(m):
    if subsimplices[i].tolist() in segments:
        input_current.append(1)
    elif subsimplices[i].tolist()[::-1] in segments:
        input_current.append(-1)
    else:
        input_current.append(0)


#%% Multi-scale flat norm
lambda_1 = 1
x,s,norm = msfn(points, simplices, subsimplices, input_current, lambda_1)
flat_lines1 = get_flat_approx(x, points, subsimplices)

lambda_2 = 500
x,s,norm = msfn(points, simplices, subsimplices, input_current, lambda_2)
flat_lines2 = get_flat_approx(x, points, subsimplices)

lambda_3 = 5000
x,s,norm = msfn(points, simplices, subsimplices, input_current, lambda_3)
flat_lines3 = get_flat_approx(x, points, subsimplices)

lambda_4 = 50000
x,s,norm = msfn(points, simplices, subsimplices, input_current, lambda_4)
flat_lines4 = get_flat_approx(x, points, subsimplices)


#%% Plot the result of MSFN
fig = plt.figure(figsize=(40,20))
ax1 = fig.add_subplot(221)
draw_points(ax1,geom_vertices,color='blue',size=0.1,alpha=0.4,marker='o')
draw_lines(ax1,geom_segments,color='blue',width=1.0,style='solid',alpha=0.2)
draw_lines(ax1,flat_lines1,color='green',width=1.0,style='solid',alpha=1.0,directed=False)
ax1.set_title("Scale="+str(lambda_1),fontsize=45)
ax1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax2 = fig.add_subplot(222)
draw_points(ax2,geom_vertices,color='blue',size=0.1,alpha=0.4,marker='o')
draw_lines(ax2,geom_segments,color='blue',width=1.0,style='solid',alpha=0.2)
draw_lines(ax2,flat_lines2,color='green',width=1.0,style='solid',alpha=1.0,directed=False)
ax2.set_title("Scale="+str(lambda_2),fontsize=45)
ax2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax3 = fig.add_subplot(223)
draw_points(ax3,geom_vertices,color='blue',size=0.1,alpha=0.4,marker='o')
draw_lines(ax3,geom_segments,color='blue',width=1.0,style='solid',alpha=0.2)
draw_lines(ax3,flat_lines3,color='green',width=1.0,style='solid',alpha=1.0,directed=False)
ax3.set_title("Scale="+str(lambda_3),fontsize=45)
ax3.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax4 = fig.add_subplot(224)
draw_points(ax4,geom_vertices,color='blue',size=0.1,alpha=0.4,marker='o')
draw_lines(ax4,geom_segments,color='blue',width=1.0,style='solid',alpha=0.2)
draw_lines(ax4,flat_lines4,color='green',width=1.0,style='solid',alpha=1.0,directed=False)
ax4.set_title("Scale="+str(lambda_4),fontsize=45)
ax4.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)






