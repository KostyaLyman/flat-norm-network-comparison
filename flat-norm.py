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
from shapely.geometry import LineString, Point, Polygon
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

def draw_polygons(ax,polygons,color='red',alpha=1.0,label=None):
    if len(polygons) == 0:
        return ax
    if isinstance(polygons,list):
        d = {'nodes':range(len(polygons)),
             'geometry':[geom for geom in polygons]}
    elif isinstance(polygons,dict):
        d = {'nodes':range(len(polygons)),
             'geometry':[polygons[k] for k in polygons]}
    df_polygons = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_polygons.plot(ax=ax,facecolor=color,alpha=alpha,label=label)
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
def get_geometry(input_geometry):
    out_geometry = []
    for geom in input_geometry:
        geom_vertices = [Point(c) for c in geom.coords]
        for i in range(len(geom_vertices)-1):
            out_geometry.append(LineString((geom_vertices[i],
                                            geom_vertices[i+1])))
    return out_geometry

def get_combined_structure(geometry):
    vertices = []
    for geom in geometry:
        geom_vertices = [Point(c) for c in geom.coords]
        for c in geom_vertices:
            if c not in vertices:
                vertices.append(c)
    segments = []
    for geom in geometry:
        ind1 = vertices.index(Point(geom.coords[0]))
        ind2 = vertices.index(Point(geom.coords[1]))
        segments.append((ind1,ind2))
    struct = {'vertices':np.array([v.coords[0] for v in vertices]), 
              'segments':np.array(segments)}
    return struct

def get_intersections(geom1,geom2):
    points = [g1.intersection(g2) for g1 in geom1 \
              for g2 in geom2 if g1.intersects(g2)]
    return points

def get_segments(points):
    start = points[0]
    sorted_indices = np.argsort([p.distance(start) for p in points])
    sorted_points = [points[i] for i in sorted_indices]
    return [LineString((sorted_points[i],sorted_points[i+1])) \
            for i in range(len(points)-1)]

def update_segment(segments,intersections):
    seg = []
    for g in segments:
        gpt1 = Point(g.coords[0])
        gpt2 = Point(g.coords[1])
        pt_list = []
        for pt in intersections:
            if g.distance(pt) < 1e-8:
                pt_list.append(pt)
        pt_list = [gpt1] + pt_list + [gpt2]
        seg.extend(get_segments(pt_list))
    return seg

def get_current(triangle_structure,input_geometry):
    current = []
    
    vertices = triangle_structure['vertices'].tolist()
    edges = triangle_structure['edges'].tolist()
    
    for edge in edges:
        vert1 = Point(vertices[edge[0]])
        vert2 = Point(vertices[edge[1]])
        forward_geom = LineString((vert1,vert2))
        reverse_geom = LineString((vert2,vert1))
        if forward_geom in input_geometry:
            current.append(1)
        elif reverse_geom in input_geometry:
            current.append(-1)
        else:
            current.append(0)
    return np.array(current)


def get_vertseg_geometry(geometry):
    struct = get_combined_structure(geometry)
    geom_vertices = [Point(v) for v in struct['vertices'].tolist()]
    geom_segments = [LineString((struct['vertices'][c[0]],
                                 struct['vertices'][c[1]])) \
                      for c in struct['segments']]
    return geom_vertices, geom_segments


#%% Toy Example

# Input geometries
geom1 = [LineString((Point(0,0),Point(1,1))),
         LineString((Point(1,1),Point(2,0))),
         LineString((Point(2,0),Point(3,1),Point(4,0)))]
geom2 = [LineString((Point(0,0.5),Point(2,0.5),Point(4,0.5)))]

geom1 = get_geometry(geom1)
geom2 = get_geometry(geom2)

# Get intersection of geometries
pt_intersect = get_intersections(geom1, geom2)

# Update geometries with new points and segments
new_geom1 = update_segment(geom1,pt_intersect)
new_geom2 = update_segment(geom2,pt_intersect)

# Combine geometries for triangulation
struct = get_combined_structure(new_geom1 + new_geom2)

# Add vertices and segments to surround the area
vert_cords = struct['vertices']
min_cord = np.min(vert_cords,axis=0) - 0.5
max_cord = np.max(vert_cords,axis=0) + 0.5
extra_vert = [min_cord,[min_cord[0],max_cord[1]],
              max_cord,[max_cord[0],min_cord[1]],min_cord]
extra_geom = [LineString((extra_vert[i],extra_vert[i+1])) for i in range(4)]

# Structure with added segments
struct = get_combined_structure(new_geom1 + new_geom2 + extra_geom)


# Constrained triangulation
tri_struct = tr.triangulate(struct,opts='p')
edges = []
for tgl in tri_struct['triangles']:
    if ([tgl[0],tgl[1]] not in edges) and ([tgl[1],tgl[0]] not in edges):
        edges.append([tgl[0],tgl[1]])
    if ([tgl[1],tgl[2]] not in edges) and ([tgl[2],tgl[1]] not in edges):
        edges.append([tgl[1],tgl[2]])
    if ([tgl[2],tgl[0]] not in edges) and ([tgl[0],tgl[2]] not in edges):
        edges.append([tgl[2],tgl[0]])
tri_struct['edges'] = np.array(edges)


T1 = get_current(tri_struct,new_geom1)
T2 = get_current(tri_struct,new_geom2)
T = T1 - T2

lambda_1 = 0.001
x,s,norm = msfn(tri_struct['vertices'], tri_struct['triangles'], tri_struct['edges'], 
                T, lambda_1)


vertices = tri_struct['vertices']
triangles = tri_struct['triangles'][s[0]!=0]
edges = tri_struct['edges'][x[0]!=0]

geom_triangles = [Polygon(vertices[np.append(t,t[0])]) for t in triangles]
geom_edges = [LineString(vertices[e]) for e in edges]



#%% Plot the example
geom1_vertices,geom1_segments = get_vertseg_geometry(geom1)
geom2_vertices,geom2_segments = get_vertseg_geometry(geom2)
new_geom1_vertices,new_geom1_segments = get_vertseg_geometry(new_geom1)
new_geom2_vertices,new_geom2_segments = get_vertseg_geometry(new_geom2)




geom_all_vertices = [Point(v) for v in struct['vertices'].tolist()]
geom_all_segments = [LineString((struct['vertices'][c[0]],
                                 struct['vertices'][c[1]])) \
                      for c in struct['segments']]
geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                 tri_struct['vertices'][c[1]])) \
                      for c in tri_struct['edges']]



    
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
draw_points(ax1,geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom1_segments,color='red',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Ground Truth Geometry')
draw_points(ax1,geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom2_segments,color='blue',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Estimated Geometry')
ax1.legend(fontsize=15, markerscale=2)

ax2 = fig.add_subplot(222)
draw_points(ax2,new_geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
draw_lines(ax2,new_geom1_segments,color='red',width=1.0,style='solid',alpha=1.0,
            directed=False,label='Ground Truth Geometry')
draw_points(ax2,new_geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
draw_lines(ax2,new_geom2_segments,color='blue',width=1.0,style='solid',alpha=1.0,
            directed=False,label='Estimated Geometry')
ax2.legend(fontsize=15, markerscale=2)


ax3 = fig.add_subplot(223)
draw_points(ax3,geom_all_vertices,color='magenta',size=20,alpha=1.0,marker='o')
draw_lines(ax3,geom_all_segments,color='magenta',width=1.0,style='solid',alpha=1.0,
           directed=False)


ax4 = fig.add_subplot(224)
draw_points(ax4,geom_vertices,color='black',size=20,alpha=0.5,marker='o')
draw_lines(ax4,geom_subsimplices,color='black',width=0.5,style='dashed',alpha=0.5,
           directed=False)
draw_lines(ax4,geom_edges,color='green',width=3.0,style='solid',alpha=1.0,
           directed=False)
draw_polygons(ax4,geom_triangles,color='magenta',alpha=0.4,label=None)



























