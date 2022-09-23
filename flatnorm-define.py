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
from shapely.geometry import LineString, Point, Polygon, MultiLineString
import triangle as tr
import matplotlib.pyplot as plt

workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

from pyUtilslib import simpvol, boundary_matrix
from pyLPsolverlib import lp_solver
from pyDrawNetworklib import plot_input, plot_intermediate_result, plot_triangulation, plot_norm


#%% Flat norm theory
import scipy.stats as stats
import math

mu1 = 0
variance1 = 49
sigma1 = math.sqrt(variance1)
mu2 = 100
variance2 = 169
sigma2 = math.sqrt(variance2)

x = np.linspace(mu1 - 10*sigma1, mu2 + 7*sigma2, 1000 )

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.plot(x, stats.norm.pdf(x, mu1, sigma1)+stats.norm.pdf(x, mu2, sigma2))









sys.exit(0)

#%% MSFN definition
def msfn(points, simplices, subsimplices, input_current, lambda_, 
         w=[], v=[], cons=[],k=1):
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
        w = simpvol(points, subsimplices, k=k)
    if v == []:
        v = simpvol(points, simplices, k=k)
    if cons == []:
        b_matrix = boundary_matrix(simplices, subsimplices, format='coo')
        m_subsimplices_identity = sparse.identity(m_subsimplices, dtype=np.int8, format='coo')
        cons = sparse.hstack((m_subsimplices_identity, -m_subsimplices_identity, b_matrix, -b_matrix))
    
    # compute the flat norm distance
    c = np.concatenate((abs(w), abs(w), lambda_*abs(v), lambda_*abs(v))) 
    c.reshape(len(c),1)
    sol, norm = lp_solver(c, cons, input_current)
    x = (sol[0:m_subsimplices] - sol[m_subsimplices:2*m_subsimplices]).reshape((1,m_subsimplices)).astype(int)
    s = (sol[2*m_subsimplices:2*m_subsimplices+n_simplices] - sol[2*m_subsimplices+n_simplices:]).reshape(1, n_simplices).astype(int)
    return x, s, norm

#%% Functions to create data structure for triangulation
def get_geometry(geometry):
    vertices = [Point(c) for c in geometry.coords]
    return [LineString((pt1,pt2)) \
            for pt1,pt2 in zip(vertices,vertices[1:]) \
                if pt1.distance(pt2) > 1e-6]


def get_current(triangle_structure,geometry):
    current = []
    vertices = triangle_structure['vertices'].tolist()
    for edge in triangle_structure['edges'].tolist():
        vert1 = Point(vertices[edge[0]])
        vert2 = Point(vertices[edge[1]])
        forward_geom = LineString((vert1,vert2))
        reverse_geom = LineString((vert2,vert1))
        if forward_geom in geometry:
            current.append(1)
        elif reverse_geom in geometry:
            current.append(-1)
        else:
            current.append(0)
    return np.array(current)

def get_structure(geometry):
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


def get_segments(points,start):
    sorted_indices = np.argsort([p.distance(start) for p in points])
    sorted_points = [points[i] for i in sorted_indices]
    return [LineString((sorted_points[i],sorted_points[i+1])) \
            for i in range(len(points)-1)]

def update_segment(segments,vertices):
    seg = []
    for g in segments:
        pt_list = []
        for pt in vertices:
            if g.distance(Point(pt)) < 1e-8:
                pt_list.append(Point(pt))
        seg.extend(get_segments(pt_list,Point(g.coords[0])))
    return seg

def prepare_triangulation(segments1,segments2):
    # Add rectangle envelope bounding the segments
    all_lines = MultiLineString(segments1+segments2)
    rect_env = list(all_lines.minimum_rotated_rectangle.buffer(0.1).boundary.coords)
    extra_geom = [LineString((Point(rect_env[i]),Point(rect_env[i+1]))) \
                  for i in range(len(rect_env)-1)]

    # Structure with added segments
    struct = get_structure(extra_geom + segments1 + segments2)
    # print(len(extra_geom))
    return struct


def perform_triangulation(act_geom,syn_geom,adj=1):
    # Initialize dictionary
    dict_struct = {}
    
    # Prepare triangulation
    struct = prepare_triangulation(act_geom,syn_geom)
    dict_struct['intermediate'] = struct
    print("Task completed: Obtained vertices and segments for constrained triangulation")
    
    # Perform constrained triangulation
    vertices = struct['vertices']
    adj_vertices = adj*(vertices + np.column_stack([[80]*len(vertices), 
                                               [-37]*len(vertices)]))
    adj_struct = {'vertices': adj_vertices,
                  'segments':struct['segments']}
    try:
        adj_tri_struct = tr.triangulate(adj_struct,opts='ps')
        adj_tri_vertices = adj_tri_struct['vertices']
        adj_mat = np.column_stack([[80]*len(adj_tri_vertices),
                                   [-37]*len(adj_tri_vertices)])
        tri_struct = {'vertices':(adj_tri_vertices/adj) - adj_mat,
                      'segments':adj_tri_struct['segments'],
                      'triangles':adj_tri_struct['triangles']}
        
        edges = []
        for tgl in tri_struct['triangles']:
            if ([tgl[0],tgl[1]] not in edges) and ([tgl[1],tgl[0]] not in edges):
                edges.append([tgl[0],tgl[1]])
            if ([tgl[1],tgl[2]] not in edges) and ([tgl[2],tgl[1]] not in edges):
                edges.append([tgl[1],tgl[2]])
            if ([tgl[2],tgl[0]] not in edges) and ([tgl[0],tgl[2]] not in edges):
                edges.append([tgl[2],tgl[0]])
        tri_struct['edges'] = np.array(edges)
        dict_struct['triangulated'] = tri_struct
        print("Task completed: Performed triangulation on points")
        
        # update input geometries with intersecting points
        V = tri_struct['vertices'].tolist()
        dict_struct['actual'] = update_segment(act_geom, V)
        dict_struct['synthetic'] = update_segment(syn_geom, V)
        print("Task completed: Updated geometries with intersecting points")
    
    except:
        print("Failed triangulation!!!")
        dict_struct['actual'] = act_geom
        dict_struct['synthetic'] = syn_geom
        dict_struct['triangulated'] = None
    
    return dict_struct


#%% Toy Example

# Input geometries
in_geom1 = [LineString((Point(0,0),Point(1,1))),
         LineString((Point(1,1),Point(2,0))),
         LineString((Point(2,0),Point(3,1),Point(4,0)))]
in_geom2 = [LineString((Point(0,0.5),Point(2,0.5),Point(4,0.5)))]

def construct_geom(points):
    return [LineString([Point(p) for p in pt_list]) for pt_list in points]

in_geom1 = construct_geom([[(-1,0),(0,0),(0,-1),(1,-1)],[(0,0),(1,0)]])
in_geom2 = construct_geom([[(-1,-1),(1,0.5)],[(-1,-0.5),(0,0.25),(1,0.25)],
                           [(0,0.25),(-1,0.25)]])
    

geom1 = []
geom2 = []
for g in in_geom1:
    geom1.extend(get_geometry(g))
for g in in_geom2:
    geom2.extend(get_geometry(g))

D = perform_triangulation(geom1,geom2,adj=1)


if D['triangulated'] == None:
    fig_ = plot_intermediate_result(D)
    sys.exit(0)


T1 = get_current(D['triangulated'], D['actual'])
T2 = get_current(D['triangulated'], D['synthetic'])

print("Task completed: obtained current information")

T = T1 - T2

print(sum(abs(T1)))
print(len(D['actual']))
print(sum(abs(T2)))
print(len(D['synthetic']))

lambda_ = 0.001
x,s,norm = msfn(D['triangulated']['vertices'], D['triangulated']['triangles'], 
                D['triangulated']['edges'], T, lambda_,k=1)

print("The computed simplicial flat norm is:",norm)

#%% Plot the example

    
fig = plt.figure(figsize=(20,10))
# Plot 1: Plot the geometries of the pair of networks
ax1 = fig.add_subplot(221)
ax1 = plot_input(geom1,geom2,ax1)
# Plot 2: All segments and points in the pre-triangulated phase
ax2 = fig.add_subplot(222)
ax2 = plot_intermediate_result(D["intermediate"],ax2)
# Plot 3: Post-triangulation phase with currents
ax3 = fig.add_subplot(223)
ax3 = plot_triangulation(D["triangulated"],T1,T2,ax3)
# Plot 4: flat-norm computation
ax4 = fig.add_subplot(224)
ax4 = plot_norm(D["triangulated"],x,s,ax4)


#%% Plot for multiple lambdas
for lambda_ in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                2,3,4,5,6,7,8,9,10]:
    x,s,norm = msfn(D['triangulated']['vertices'], D['triangulated']['triangles'], 
                    D['triangulated']['edges'], T, lambda_,k=1)
    print("The computed simplicial flat norm is:",norm)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax1 = plot_triangulation(D["triangulated"],T1,T2,ax1)
    ax2 = fig.add_subplot(122)
    ax2 = plot_norm(D["triangulated"],x,s,ax2)























