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
from shapely.geometry import LineString, Point, Polygon, MultiPoint, MultiLineString
import matplotlib.pyplot as plt
import triangle as tr
import networkx as nx


workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
actpath = workpath + "/input/actual/"
synpath = workpath + "/input/primnet/"
outpath = workpath + "/out/"

from pyUtilslib import simpvol, boundary_matrix
from pyLPsolverlib import lp_solver
from pyExtractDatalib import GetDistNet
from pyGeometrylib import geodist
from pyDrawNetworklib import plot_norm, plot_intermediate_result, plot_input, plot_triangulation


#%% SYnthetic network notes
# hethwood - portion of 150724
# mcbryde - portions of 150724 and 150692
# sublist = [147793, 148717, 148718, 148719, 148720, 148721, 148723,
#        150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#        150727, 150728]

#%% Output functions

def out_struct(struct,path):
    vert_list = struct['vertices'].tolist()
    edge_list = struct['segments'].tolist()
    vert_data = '\n'.join(['\t'.join([str(v) for v in vert]) for vert in vert_list])
    edge_data = '\n'.join(['\t'.join([str(e) for e in edge]) for edge in edge_list])
    
    with open(path + 'in-vertices.txt','w') as f:
        f.write(vert_data)
    with open(path + 'in-edges.txt','w') as f:
        f.write(edge_data)
    return

def out_geom(geometry,path):
    data = '\n'.join(['\t'.join([str(x) for x in g.coords]) for g in geometry])
    with open(path + 'geometry.txt','w') as f:
        f.write(data)
    return

def out_tristruct(struct,path):
    vert_list = struct['vertices'].tolist()
    edge_list = struct['segments'].tolist()
    trng_list = struct['triangles'].tolist()
    tedg_list = struct['edges'].tolist()
    vert_data = '\n'.join(['\t'.join([str(v) for v in vert]) for vert in vert_list])
    edge_data = '\n'.join(['\t'.join([str(e) for e in edge]) for edge in edge_list])
    trng_data = '\n'.join(['\t'.join([str(t) for t in trng]) for trng in trng_list])
    tedg_data = '\n'.join(['\t'.join([str(d) for d in tedg]) for tedg in tedg_list])
    
    with open(path + 'out-vertices.txt','w') as f:
        f.write(vert_data)
    with open(path + 'out-edges.txt','w') as f:
        f.write(edge_data)
    with open(path + 'out-triangles.txt','w') as f:
        f.write(trng_data)
    with open(path + 'out-subsimplices.txt','w') as f:
        f.write(tedg_data)
    return

#%% epsilon neighborhood
def sample_vertices(pt,e,num=1):
    dev = np.random.random((num,2))
    r = e * np.sqrt(dev[:,0])
    theta = 2 * np.pi * dev[:,1]
    x = pt.x + (r * np.cos(theta))
    y = pt.y + (r * np.sin(theta))
    return [Point(x[i],y[i]) for i in range(num)]

def sample_geometries(geometry,eps,num=1):
    struct = get_structure(geometry)
    vertices = struct['vertices']
    segments = struct['segments']
    
    vert_samples = [sample_vertices(Point(v),eps,num) \
                    for v in vertices.tolist()]
    geom_samples = [[LineString((vert_samples[c[0]][i],vert_samples[c[1]][i])) \
                     for c in segments] for i in range(num)]
    return [[vert[i] for vert in vert_samples] for i in range(num)], geom_samples




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


#%% New functions
def get_geometry(geometry):
    vertices = [Point(c) for c in geometry.coords]
    return [LineString((pt1,pt2)) \
            for pt1,pt2 in zip(vertices,vertices[1:]) \
                if geodist(pt1,pt2) > 1e-6]


def get_current(triangle_structure,geometry):
    current = []
    vertices = triangle_structure['vertices'].tolist()
    for edge in triangle_structure['edges'].tolist():
        vert1 = Point(vertices[edge[0]])
        vert2 = Point(vertices[edge[1]])
        forward_geom = LineString((vert1,vert2))
        reverse_geom = LineString((vert2,vert1))
        for_eq = [forward_geom.almost_equals(geom) for geom in geometry]
        rev_eq = [reverse_geom.almost_equals(geom) for geom in geometry]
        if sum(for_eq)>0:
            current.append(1)
        elif sum(rev_eq)>0:
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
    rect_env = list(all_lines.minimum_rotated_rectangle.buffer(1e-4).boundary.coords)
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




#%% Extract actual network data and direct the geometries from left to right
areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
area = 'mcbryde'
# area = 'hethwood'

# Get dictionary of node geometries and edgelist
busgeom = {}
edgelist = []
edgegeom = {}

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


sorted_edges = list(nx.dfs_edges(act_graph, source=left_node))

all_sorted_act_geom = []
for e in sorted_edges:
    edge_id1 = '_'.join([str(x) for x in list(e)])
    edge_id2 = '_'.join([str(x) for x in list(e)[::-1]])
    d = df_lines.loc[(df_lines.ID == edge_id1) | (df_lines.ID == edge_id2)]
    all_sorted_act_geom.extend(get_geometry(d["geometry"][d.index[0]]))

# sorted_act_geom = all_sorted_act_geom
sorted_act_geom = all_sorted_act_geom
print("Task completed: Actual network geometry sorted")

#%% Extract the synthetic network data for the region
sublist = [150692,150724]
synth = GetDistNet(synpath,sublist)


# act_lines = MultiLineString(sorted_act_geom)
# rect_env = act_lines.minimum_rotated_rectangle.buffer(5e-4)
act_vertices = MultiPoint([Point(g) \
                           for geom in sorted_act_geom for g in geom.coords])
hull = act_vertices.convex_hull.buffer(5e-4)


nodelist = [n for n in synth.nodes \
            if Point(synth.nodes[n]['cord']).within(hull) \
                and synth.nodes[n]['label'] != 'H']


# sort the edges to store data
sorted_syn_geom = []
syn_graph = nx.subgraph(synth,nodelist)
for comp in list(nx.connected_components(syn_graph)):
    x_cord = [synth.nodes[n]['cord'][0] for n in comp]
    l_node = list(comp)[np.argmin(x_cord)]
    sorted_syn_edges = list(nx.dfs_edges(syn_graph, source=l_node))
    for e in sorted_syn_edges:
        sorted_syn_geom.extend(get_geometry(synth.edges[e]['geometry']))

print("Task completed: Synthetic network geometry sorted")


#%% Delete manually
# remlist = [18,63,64,65,66,75,76,84,85,86,87,88]
# remlist = [18]
# remlist = []
# sorted_syn_geom = [g for i,g in enumerate(sorted_syn_geom) if i not in remlist]

# for j in range(len(sorted_syn_geom)):
#     new_sorted_syn_geom = [g for i,g in enumerate(sorted_syn_geom) if i != j]
#     fig = plot_input(sorted_act_geom,new_sorted_syn_geom)
#     filename = 'temp-'+str(j)
#     fig.savefig(workpath+'/figs/tmp/'+filename+'.png',bbox_inches='tight')
# sys.exit(0)

#%% Plot the network

# num_s = 5
# eps = 1e-4

# act_vert_samp, act_seg_samp = sample_geometries(sorted_act_geom,eps,num=num_s)
# syn_vert_samp, syn_seg_samp = sample_geometries(sorted_syn_geom,eps,num=num_s)

# fig = plt.figure(figsize=(20,20))
# ax = fig.add_subplot(111)


# draw_lines(ax,sorted_act_geom,color='red',label='actual')
# draw_lines(ax,sorted_syn_geom,color='blue',label='synthetic')
# draw_lines(ax,extra_geom,color='black',label='extra')

# for i in range(num_s):
#     draw_points(ax,act_vert_samp[i],color='red',size=10,alpha=1.0,marker='o')
#     draw_lines(ax,act_seg_samp[i],color='red',width=1.0,style='dashed',alpha=0.5,
#                 directed=False,label='Sample '+str(i+1) + ' Geometry')
# ax.legend(fontsize=20, markerscale=3)

# sys.exit(0)

#%% Flat norm computation
D = perform_triangulation(sorted_act_geom,sorted_syn_geom,adj=1000)


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


fig = plt.figure(figsize=(40,60))

# Plot 1: Plot the geometries of the pair of networks
ax1 = fig.add_subplot(311)
ax1 = plot_input(D["actual"],D["synthetic"],ax1)
# Plot 2: All segments and points in the pre-triangulated phase
ax2 = fig.add_subplot(312)
ax2 = plot_intermediate_result(D["intermediate"],ax2)
# Plot 3: Post-triangulation phase with currents
ax3 = fig.add_subplot(313)
ax3 = plot_triangulation(D["triangulated"],T1,T2,ax3)



sys.exit(0)
#%% Perform flat norm computation
# lambda_ = 1000
for lambda_ in np.linspace(1000,100000,10):
    x,s,norm = msfn(D['triangulated']['vertices'], D['triangulated']['triangles'], 
                    D['triangulated']['edges'], T, lambda_,k=np.pi/(180.0))
    
    print("The computed simplicial flat norm is:",norm)
    # Show results
    fig = plt.figure(figsize=(60,40))
    ax = fig.add_subplot(111)
    plot_norm(D["triangulated"],x,s,ax)
    ax.set_title("Flat norm scale, lambda = "+str(lambda_), fontsize=30)
    filename = area+'-lambda-'+str(lambda_)
    fig.savefig(workpath+'/figs/'+filename+'.png',bbox_inches='tight')
    # sys.exit(0)





