# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:27:55 2022

@author: rm5nz
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
from shapely.geometry import LineString, Point
import triangle as tr
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

#%% Functions to create data structure for triangulation
def structure(geometry):
    vertices = []
    segments = []
    for geom in geometry:
        geom_vertices = [Point(c) for c in geom.coords]
        for c in geom_vertices:
            if c not in vertices:
                vertices.append(c)
        for i in range(len(geom_vertices)-1):
            ind1 = vertices.index(geom_vertices[i])
            ind2 = vertices.index(geom_vertices[i+1])
            segments.append((ind1,ind2))
    struct = {'vertices':np.array([v.coords[0] for v in vertices]), 
              'segments':np.array(segments)}
    return struct

#%% Constrained triangulation
def constrained_triangulation(input_struct):
    t = tr.triangulate(input_struct,'pc')
    # Get the edges
    subsimplices = []
    for tgl in t['triangles']:
        if ([tgl[0],tgl[1]] not in subsimplices) and ([tgl[1],tgl[0]] not in subsimplices):
            subsimplices.append([tgl[0],tgl[1]])
        if ([tgl[1],tgl[2]] not in subsimplices) and ([tgl[2],tgl[1]] not in subsimplices):
            subsimplices.append([tgl[1],tgl[2]])
        if ([tgl[2],tgl[0]] not in subsimplices) and ([tgl[0],tgl[2]] not in subsimplices):
            subsimplices.append([tgl[2],tgl[0]])
    
    output_struct = {'vertices':t['vertices'],'simplices':t['triangles'],
                     'subsimplices':np.array(subsimplices)}
    return output_struct

def update_segments(org_struct,tri_struct):
    # Get the input segments
    in_vertices = org_struct['vertices']
    in_segments = org_struct['segments']
    geom_segments = [LineString((in_vertices[c[0]],in_vertices[c[1]])) \
                     for c in in_segments]
    segments = []
    for seg in tri_struct['subsimplices'].tolist():
        pts_seg = tri_struct['vertices'][seg].tolist()
        geom_seg1 = LineString([Point(pts_seg[0]),Point(pts_seg[1])])
        geom_seg2 = LineString([Point(pts_seg[1]),Point(pts_seg[0])])
        if (geom_seg1 in geom_segments) or (geom_seg2 in geom_segments):
            segments.append(seg)
    return

def get_current(struct):
    current = []
    m = struct['subsimplices'].shape[0]
    for i in range(m):
        if struct['subsimplices'][i].tolist() in struct['segments']:
            current.append(1)
        elif struct['subsimplices'][i].tolist()[::-1] in struct['segments']:
            current.append(-1)
        else:
            current.append(0)
    return current

#%% epsilon neighborhood
def sample_vertices(pt,e,num=1):
    dev = np.random.random((num,2))
    r = e * np.sqrt(dev[:,0])
    theta = 2 * np.pi * dev[:,1]
    x = pt.x + (r * np.cos(theta))
    y = pt.y + (r * np.sin(theta))
    return [Point(x[i],y[i]) for i in range(num)]

def sample_geometries(geometry,eps,num=1):
    struct = structure(geometry)
    vertices = struct['vertices']
    segments = struct['segments']
    
    vert_samples = [sample_vertices(Point(v),eps,num) \
                    for v in vertices.tolist()]
    geom_samples = [[LineString((vert_samples[c[0]][i],vert_samples[c[1]][i])) \
                     for c in segments] for i in range(num)]
    return [[vert[i] for vert in vert_samples] for i in range(num)], geom_samples


#%% Example
np.random.seed(121)

geom1 = [LineString((Point(0,0),Point(1,0))),
         LineString((Point(1,0),Point(2,0))),
         LineString((Point(1,0),Point(2,1)))]

geom2 = [LineString((Point(0,-0.5),Point(1,0))),
         LineString((Point(1,0),Point(2,0.5))),
         LineString((Point(1,0),Point(2,1.5)))]


struct1 = structure(geom1)
struct2 = structure(geom2)

geom1_vertices = [Point(v) for v in struct1['vertices']]
geom2_vertices = [Point(v) for v in struct2['vertices']]

geom1_segments = [LineString((struct1['vertices'][c[0]],
                              struct1['vertices'][c[1]])) \
                  for c in struct1['segments']]
geom2_segments = [LineString((struct2['vertices'][c[0]],
                              struct2['vertices'][c[1]])) \
                  for c in struct2['segments']]


num_samples = 5


#%% Plot the geometries
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
draw_points(ax1,geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom1_segments,color='red',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Ground Truth Geometry')
draw_points(ax1,geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom2_segments,color='blue',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Estimated Geometry')
vert_samp, seg_samp = sample_geometries(geom2,0.1,num=num_samples)
for i in range(num_samples):
    draw_points(ax1,vert_samp[i],color='blue',size=10,alpha=1.0,marker='o')
    draw_lines(ax1,seg_samp[i],color='blue',width=0.5,style='dashed',alpha=1.0,
               directed=False,label='Sample '+str(i+1) + ' Geometry')
ax1.legend(fontsize=20, markerscale=3)

ax2 = fig.add_subplot(122)
draw_points(ax2,geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
draw_lines(ax2,geom1_segments,color='red',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Ground Truth Geometry')
draw_points(ax2,geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
draw_lines(ax2,geom2_segments,color='blue',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Estimated Geometry')
vert_samp, seg_samp = sample_geometries(geom1,0.1,num=num_samples)
for i in range(num_samples):
    draw_points(ax2,vert_samp[i],color='red',size=10,alpha=1.0,marker='o')
    draw_lines(ax2,seg_samp[i],color='red',width=0.5,style='dashed',alpha=1.0,
               directed=False,label='Sample '+str(i+1) + ' Geometry')
ax2.legend(fontsize=20, markerscale=3)

fig.savefig(workpath+'/figs/graph-sample.png',bbox_inches='tight')










































