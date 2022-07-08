# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:23:40 2022

Author: Rounak Meyur
"""

from __future__ import absolute_import

import sys,os
import numpy as np
from scipy import sparse

workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

from pyUtilslib import simpvol, boundary_matrix
from pyLPsolverlib import lp_solver

#%% Plot functions
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint
import matplotlib.pyplot as plt


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

#%%

lines = [LineString((Point(2,0),Point(1,0))),
         LineString((Point(1,0),Point(0,0))),
         LineString((Point(0,0),Point(1,1))),
         LineString((Point(1,1),Point(2,1))),
         LineString((Point(1,2),Point(1,1))),
         LineString((Point(1,1),Point(1,0))),
         LineString((Point(1,0),Point(2,1)))]


pts_check = [Point(0,0),Point(1,0),Point(1,1),Point(2,0),Point(2,1),Point(1,2)]

pts = []
for line_geom in lines:
    pt_geom = line_geom.interpolate(0.5)
    pts.append.extend()



sys.exit(0)

chain = [[0,1],[0,2],[1,2],[3,1],[1,4],[4,2],[3,4],[5,2],[5,4]]
geom_subsimplices = [LineString((pts[c[0]],pts[c[1]])) for c in chain]


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
draw_points(ax1,pts,color='black',size=50,alpha=1.0,marker='o')
draw_lines(ax1,geom_subsimplices,color='black',width=2.0,style='dashed',alpha=1.0,
           directed=True)
ax1.set_title("Simplicial complexes",fontsize=15)
ax2 = fig.add_subplot(212)
draw_points(ax2,pts,color='blue',size=50,alpha=1.0,marker='o')
draw_lines(ax2,lines,color='blue',width=2.0,style='solid',alpha=1.0,directed=True)
ax2.set_title("Input current",fontsize=15)


#%% Input geometry
points = np.array([pt.coords[0] for pt in pts])
# 2-simplices: triangles
simplices = np.array([[0,2,1],[1,2,4],[1,4,3],[2,4,5]])
# 1-simplices: edges
subsimplices = np.array(chain)

# input current
input_edges = [[order[i],order[i+1]] for i in range(len(order)-1)]
input_edges += [[order_[i],order_[i+1]] for i in range(len(order_)-1)]
input_current = []
m = subsimplices.shape[0]
for i in range(m):
    if subsimplices[i].tolist() in input_edges:
        input_current.append(1)
    elif subsimplices[i].tolist()[::-1] in input_edges:
        input_current.append(-1)
    else:
        input_current.append(0)


#%% Multi-scale flat norm
lambda_1 = 1
x,s,norm = msfn(points, simplices, subsimplices, input_current, lambda_1)
flat_lines1 = get_flat_approx(x, points, subsimplices)

lambda_2 = 5
x,s,norm = msfn(points, simplices, subsimplices, input_current, lambda_2)
flat_lines2 = get_flat_approx(x, points, subsimplices)


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
draw_points(ax1,pts,color='blue',size=50,alpha=0.4,marker='o')
draw_lines(ax1,lines,color='blue',width=2.0,style='solid',alpha=0.2)
draw_lines(ax1,flat_lines1,color='green',width=2.0,style='solid',alpha=1.0,directed=True)
ax1.set_title("Scale="+str(lambda_1),fontsize=15)
ax2 = fig.add_subplot(212)
draw_points(ax2,pts,color='blue',size=50,alpha=0.4,marker='o')
draw_lines(ax2,lines,color='blue',width=2.0,style='solid',alpha=0.2)
draw_lines(ax2,flat_lines2,color='green',width=2.0,style='solid',alpha=1.0,directed=True)
ax2.set_title("Scale="+str(lambda_2),fontsize=15)


#%% Get set of points for drawing the flat norm manifold
hull = MultiPoint(pts).convex_hull.boundary




