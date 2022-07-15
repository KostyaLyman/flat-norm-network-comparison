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
from shapely.geometry import LineString, Point, Polygon
import matplotlib.pyplot as plt
import triangle as tr
import networkx as nx


workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
actpath = workpath + "/input/actual/"
synpath = workpath + "/input/primnet/"

from pyUtilslib import simpvol_geod, boundary_matrix
from pyLPsolverlib import lp_solver
from pyExtractDatalib import GetDistNet


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


#%% Functions to create data structure for triangulation
def get_combined_structure(geometry):
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

def get_current(triangle_structure,geometry):
    current = []
    for edge in triangle_structure['edges'].tolist():
        vert1 = Point(triangle_structure['vertices'][edge[0]])
        vert2 = Point(triangle_structure['vertices'][edge[1]])
        forward_geom = LineString((vert1,vert2))
        reverse_geom = LineString((vert2,vert1))
        if forward_geom in geometry:
            current.append(1)
        elif reverse_geom in geometry:
            current.append(-1)
        else:
            current.append(0)
    return np.array(current)



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
        w = simpvol_geod(points, subsimplices)
    if v == []:
        v = simpvol_geod(points, simplices)
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


#%% Extract actual network data and direct the geometries from left to right
areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
area = 'hethwood'

# Create actual graph network
df_lines = gpd.read_file(actpath+area+'/'+area+'_edges.shp')
df_buses = gpd.read_file(actpath+area+'/'+area+'_nodes.shp')


# Get dictionary of node geometries and edgelist
edgelist = [tuple([int(x) for x in e.split('_')]) for e in df_lines['ID']]

# Create actual graph network
act_graph = nx.Graph()
act_graph.add_edges_from(edgelist)


act_nodes = act_graph.nodes
busgeom = {df_buses['id'][i]:df_buses['geometry'][i] \
           for i in range(len(df_buses))}
x_bus = [busgeom[n].x for n in act_nodes]
y_bus = [busgeom[n].y for n in act_nodes]
left_node = list(act_nodes)[np.argmin(x_bus)]

buffer = 1e-4
left = min(x_bus)-buffer
right = max(x_bus)+buffer
bottom = min(y_bus)-buffer
top = max(y_bus)+buffer


sorted_edges = list(nx.dfs_edges(act_graph, source=left_node))
sorted_act_geom = []
for e in sorted_edges:
    edge_id1 = '_'.join([str(x) for x in list(e)])
    edge_id2 = '_'.join([str(x) for x in list(e)[::-1]])
    d = df_lines.loc[(df_lines.ID == edge_id1) | (df_lines.ID == edge_id2)]
    geom = d["geometry"][d.index[0]]
    sorted_act_geom.append(geom)


#%% Extract synthetic network data
sublist = [150722, 150723, 150724, 150725, 150726, 150727, 150728]
synth = GetDistNet(synpath,sublist)

nodelist = [n for n in synth.nodes \
            if left<=synth.nodes[n]['cord'][0]<=right \
             and bottom<=synth.nodes[n]['cord'][1]<=top \
                 and synth.nodes[n]['label'] != 'H']

sorted_syn_geom = []
syn_graph = nx.subgraph(synth,nodelist)
for comp in nx.connected_components(syn_graph):
    x_cord = [synth.nodes[n]['cord'][0] for n in comp]
    l_node = list(comp)[np.argmin(x_cord)]
    sorted_syn_edges = list(nx.dfs_edges(syn_graph, source=l_node))
    for e in sorted_syn_edges:
        sorted_syn_geom.append(synth.edges[e]['geometry'])


#%% Prepare for triangulation
act_struct = get_combined_structure(sorted_act_geom)
syn_struct = get_combined_structure(sorted_syn_geom)

# Get intersection of geometries
act_geom = [LineString((act_struct['vertices'][s[0]],
                        act_struct['vertices'][s[1]])) \
            for s in act_struct['segments']]
syn_geom = [LineString((syn_struct['vertices'][s[0]],
                        syn_struct['vertices'][s[1]])) \
            for s in syn_struct['segments']]
pt_intersect = get_intersections(act_geom, syn_geom)

# Update geometries with new points and segments
new_geom1 = update_segment(act_geom,pt_intersect)
new_geom2 = update_segment(syn_geom,pt_intersect)

# Combine geometries for triangulation
struct = get_combined_structure(new_geom1 + new_geom2)

# Add vertices and segments to surround the area
vert_cords = struct['vertices']
min_cord = np.min(vert_cords,axis=0) - buffer
max_cord = np.max(vert_cords,axis=0) + buffer
extra_vert = [min_cord,[min_cord[0],max_cord[1]],
              max_cord,[max_cord[0],min_cord[1]],min_cord]
extra_geom = [LineString((extra_vert[i],extra_vert[i+1])) for i in range(4)]

# Structure with added segments
struct = get_combined_structure(new_geom1 + new_geom2 + extra_geom)

#%% Constrained triangulation and flat norm computation
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

#%% Flat norm computation
T1 = get_current(tri_struct, new_geom1)
T2 = get_current(tri_struct, new_geom2)

T = T1 - T2

lambda_1 = 0.001
x,s,norm = msfn(tri_struct['vertices'], tri_struct['triangles'], tri_struct['edges'], 
                T, lambda_1)


vertices = tri_struct['vertices']
triangles = tri_struct['triangles'][s[0]!=0]
edges = tri_struct['edges'][x[0]!=0]

geom_triangles = [Polygon(vertices[np.append(t,t[0])]) for t in triangles]
geom_edges = [LineString(vertices[e]) for e in edges]



#%% Get the geometries
geom1_vertices = [Point(v) for v in act_struct['vertices'].tolist()]
geom2_vertices = [Point(v) for v in syn_struct['vertices'].tolist()]
geom1_segments = [LineString((act_struct['vertices'][c[0]],
                              act_struct['vertices'][c[1]])) \
                  for c in act_struct['segments']]
geom2_segments = [LineString((syn_struct['vertices'][c[0]],
                              syn_struct['vertices'][c[1]])) \
                  for c in syn_struct['segments']]


    
#%% Plot the geometries
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(111)
draw_points(ax1,geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom1_segments,color='red',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Actual Network')
draw_points(ax1,geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom2_segments,color='blue',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Synthetic Network')
ax1.legend(fontsize=20, markerscale=3)
ax1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
