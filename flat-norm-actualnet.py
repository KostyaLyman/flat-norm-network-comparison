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

from pyUtilslib import simpvol, boundary_matrix
from pyLPsolverlib import lp_solver
from pyExtractDatalib import GetDistNet
from pyGeometrylib import geodist


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
def get_geometry(input_geometry):
    out_geometry = []
    for geom in input_geometry:
        geom_vertices = [Point(c) for c in geom.coords]
        for i in range(len(geom_vertices)-1):
            pt1 = geom_vertices[i]
            pt2 = geom_vertices[i+1]
            if geodist(pt1,pt2) > 1e-6:
                out_geometry.append(LineString((pt1,pt2)))
    return out_geometry

def filter_out(geometry):
    out_geometry = []
    for i,i_geom in enumerate(geometry):
        if len(out_geometry) == 0:
            out_geometry.append(i_geom)
        else:
            flag = 0
            for j,j_geom in enumerate(out_geometry):
                int_geom = i_geom.intersection(j_geom)
                if i_geom.intersects(j_geom) and isinstance(int_geom, LineString):
                    print(i,j)
                    flag = 1
                    break
            if flag == 0:
                out_geometry.append(i_geom)
    return out_geometry

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

def get_vertseg_geometry(struct):
    if isinstance(struct,list):
        struct = get_structure(struct)
    geom_vertices = [Point(v) for v in struct['vertices'].tolist()]
    geom_segments = [LineString((struct['vertices'][c[0]],
                                 struct['vertices'][c[1]])) \
                      for c in struct['segments']]
    return geom_vertices, geom_segments



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


#%% Extract actual network data and direct the geometries from left to right
areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
area = 'hethwood'


# Get dictionary of node geometries and edgelist
busgeom = {}
edgelist = []
edgegeom = {}
for area in ['patrick_henry']:
    df_lines = gpd.read_file(actpath+area+'/'+area+'_edges.shp')
    df_buses = gpd.read_file(actpath+area+'/'+area+'_nodes.shp')
    
    # add edges to edgelist
    for i in range(len(df_lines)):
        edge = tuple([int(x) for x in df_lines['ID'][i].split('_')])
        edgelist.append(edge)
        edgegeom[edge] = df_lines['geometry'][i]
    # add bus geometries
    for i in range(len(df_buses)):
        busgeom[int(df_buses['id'][i])] = df_buses['geometry'][i]


# Create actual graph network
act_graph = nx.Graph()
act_graph.add_edges_from(edgelist)


act_nodes = act_graph.nodes
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


print("Task completed: Actual network geometry sorted")

#%% Extract synthetic network data
# sublist = [147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]
sublist = [150724]
synth = GetDistNet(synpath,sublist)

nodelist = [n for n in synth.nodes \
            if left<=synth.nodes[n]['cord'][0]<=right \
             and bottom<=synth.nodes[n]['cord'][1]<=top \
                 and synth.nodes[n]['label'] != 'H']

sorted_syn_geom = []
syn_graph = nx.subgraph(synth,nodelist)
for comp in list(nx.connected_components(syn_graph)):
    x_cord = [synth.nodes[n]['cord'][0] for n in comp]
    l_node = list(comp)[np.argmin(x_cord)]
    sorted_syn_edges = list(nx.dfs_edges(syn_graph, source=l_node))
    for e in sorted_syn_edges:
        sorted_syn_geom.append(synth.edges[e]['geometry'])


print("Task completed: Synthetic network geometry sorted")

#%% Prepare for triangulation
act_geom = get_geometry(sorted_act_geom)
syn_geom = get_geometry(sorted_syn_geom)

pt_intersect = get_intersections(act_geom, syn_geom)
print("Task completed: Intersection of network geometries")

# Update geometries with new points and segments
new_geom_act = update_segment(act_geom,pt_intersect)
new_geom_syn = update_segment(syn_geom,pt_intersect)
# remlist = [76, 468, 851, 1198, 1557, 1577, 1630]
# new_geom2 = [g for i,g in enumerate(new_geom2) if i not in remlist]
print("Task completed: Updated geometries with intersecting points")

# Add vertices and segments to surround the area
buffer = 1e-3
left = min(x_bus)-buffer
right = max(x_bus)+buffer
bottom = min(y_bus)-buffer
top = max(y_bus)+buffer
extra_vert = [[left,bottom],[left,top],[right,top],[right,bottom],[left,bottom]]
extra_geom = [LineString((extra_vert[i],extra_vert[i+1])) for i in range(4)]

#%% Create a dummy for checking purpose
# syn_struct = get_structure(extra_geom + new_geom_syn + new_geom_act)


#%% Check for bug in triangulation

# remlist = [80,472,855,1202,1561,1581,1634]
# remlist = []

# check_struct = {}
# check_struct['vertices'] = syn_struct['vertices']
# check_struct['segments'] = np.array([seg for i,seg in enumerate(syn_struct['segments'].tolist()) \
#                                       if i not in remlist])

# # Triangulate the structure
# tri_struct = tr.triangulate(check_struct,opts='p')

# # Plot figure
# geom_syn_vertices,geom_syn_segments = get_vertseg_geometry(check_struct)

# fig = plt.figure(figsize=(15,40))
# ax1 = fig.add_subplot(111)
# # draw_points(ax1,geom_syn_vertices,color='blue',size=1,alpha=1.0,marker='o')
# draw_lines(ax1,geom_syn_segments,color='blue',width=2.0,style='solid',alpha=1.0,
#             directed=False)
# ax1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)




# sys.exit(0)
# remlist = [386,557]

# Structure with added segments
struct = get_structure(extra_geom + new_geom_syn + new_geom_act)
remlist = []

struct['segments'] = np.array([seg \
                               for i,seg in enumerate(struct['segments'].tolist()) \
                               if i not in remlist])
print("Task completed: Obtained vertices and segments for constrained triangulation")


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
print("Task completed: Performed triangulation on points")

#%% Flat norm computation
T1 = get_current(tri_struct, sorted_act_geom)
T2 = get_current(tri_struct, sorted_syn_geom)

print("Task completed: obtained current information")

T = T1 - T2

lambda_1 = 0.001
x,s,norm = msfn(tri_struct['vertices'], tri_struct['triangles'], tri_struct['edges'], 
                T, lambda_1)


vertices = tri_struct['vertices']
triangles = tri_struct['triangles'][s[0]!=0]
edges = tri_struct['edges'][x[0]!=0]

geom_triangles = [Polygon(vertices[np.append(t,t[0])]) for t in triangles]
geom_edges = [LineString(vertices[e]) for e in edges]

# sys.exit(0)

#%% Get the geometries

geom1_vertices,geom1_segments = get_vertseg_geometry(act_geom)
geom2_vertices,geom2_segments = get_vertseg_geometry(syn_geom)

geom_all_vertices = [Point(v) for v in struct['vertices'].tolist()]
geom_all_segments = [LineString((struct['vertices'][c[0]],
                                 struct['vertices'][c[1]])) \
                      for c in struct['segments']]
geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                 tri_struct['vertices'][c[1]])) \
                      for c in tri_struct['edges']]



    
#%% Plot the geometries
fig = plt.figure(figsize=(30,40))
ax1 = fig.add_subplot(121)
draw_points(ax1,geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom1_segments,color='red',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Actual Network')
draw_points(ax1,geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
draw_lines(ax1,geom2_segments,color='blue',width=1.0,style='solid',alpha=1.0,
           directed=False,label='Synthetic Network')
ax1.legend(fontsize=20, markerscale=3)
ax1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

ax2 = fig.add_subplot(122)
draw_points(ax2,geom_all_vertices,color='magenta',size=20,alpha=1.0,marker='o')
draw_lines(ax2,geom_all_segments,color='magenta',width=1.0,style='solid',alpha=1.0,
           directed=False)
ax2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

