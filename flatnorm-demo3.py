# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:23:40 2022

Author: Rounak Meyur

Description: Demonstration of how varying the scale parameter of simplicial 
flat norm alters the computed norm. The test is done for actual and synthetic
distribution networks in Blacksburg. 

The norm is computed with the two components separately: subsimplicial norm and
simplicial norm. These data are stored in a .txt file for multiple scale values.
"""

from __future__ import absolute_import

import sys,os
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint
import matplotlib.pyplot as plt
import networkx as nx


workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
actpath = workpath + "/input/actual/"
synpath = workpath + "/input/primnet/"
outpath = workpath + "/out/"


from pyExtractDatalib import GetDistNet
from pyFlatNormlib import get_geometry, get_current, msfn, perform_triangulation
from pyDrawNetworklib import plot_norm, plot_failed_triangulation

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

def sample_geometries(geometry,eps,num=1):
    struct = get_structure(geometry)
    vertices = struct['vertices']
    segments = struct['segments']
    
    vert_samples = [sample_vertices(Point(v),eps,num) \
                    for v in vertices.tolist()]
    geom_samples = [[LineString((vert_samples[c[0]][i],vert_samples[c[1]][i])) \
                     for c in segments] for i in range(num)]
    return [[vert[i] for vert in vert_samples] for i in range(num)], geom_samples



#%% Extract actual network data and direct the geometries from left to right
# areas = {'patrick_henry':194,'mcbryde':9001,'hethwood':7001}
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



#%% Flat norm computation
D = perform_triangulation(sorted_act_geom,sorted_syn_geom,adj=1000)


if D['triangulated'] == None:
    plot_failed_triangulation(D)
    sys.exit(0)


T1 = get_current(D['triangulated'], D['actual'])
T2 = get_current(D['triangulated'], D['synthetic'])

print("Task completed: obtained current information")

T = T1 - T2

print(sum(abs(T1)))
print(len(D['actual']))
print(sum(abs(T2)))
print(len(D['synthetic']))  


#%% Flat norm computation for multiple lambda
for lambda_ in np.linspace(1000,100000,100):
    x,s,norm,norm1,norm2 = msfn(D['triangulated']['vertices'],
                                D['triangulated']['triangles'], 
                                D['triangulated']['edges'], T, 
                                lambda_,k=np.pi/(180.0))
    
    # Show results for every 10 computation
    if int(lambda_)%10000 == 0:
        fig = plt.figure(figsize=(60,40))
        ax = fig.add_subplot(111)
        plot_norm(D["triangulated"],x,s,ax)
        ax.set_title("Scale, lambda = "+str(int(lambda_)) + \
                     ", Computed flat norm = "+str(norm), fontsize=60)
        filename = area+'-lambda-'+str(lambda_)
        fig.savefig(workpath+'/figs/'+filename+'.png',bbox_inches='tight')
    
    # Save the norm results
    with open(outpath+"flat-norm.txt",'a') as f:
        data = "\t".join([str(x) for x in [int(lambda_),norm,
                                           norm1,norm2]]) + "\n"
        f.write(data)
    


