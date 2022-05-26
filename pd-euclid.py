# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:02:08 2022

Author: Rounak Meyur
"""

import sys,os
import numpy as np
import networkx as nx
from shapely.geometry import LineString,Point
import geopandas as gpd
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic


#%% Classes
class Link(LineString):
    """
    Derived class from Shapely LineString to compute metric distance based on 
    geographical coordinates over geometric coordinates.
    """
    def __init__(self,line_geom):
        """
        """
        super().__init__(line_geom)
        self.geod_length = self.__length()
        return
    
    
    def __length(self):
        '''
        Computes the geographical length in meters between the ends of the link.
        '''
        if self.geom_type != 'LineString':
            print("Cannot compute length!!!")
            return None
        # Compute great circle distance
        geod = Geodesic.WGS84
        length = 0.0
        for i in range(len(list(self.coords))-1):
            lon1,lon2 = self.xy[0][i:i+2]
            lat1,lat2 = self.xy[1][i:i+2]
            length += geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        return length
    
    def InterpolatePoints(self,sep=20):
        """
        """
        points = []
        length = self.geod_length
        for i in np.arange(0,length,sep):
            x,y = self.interpolate(i/length,normalized=True).xy
            xy = (x[0],y[0])
            points.append(Point(xy))
        if len(points)==0: 
            points.append(Point((self.xy[0][0],self.xy[1][0])))
        return points


#%% Test geometry

A_pts = [(0,0.125),(2,0.3),(5,-0.125),(8,-0.125),(9,0)]
B_pts = [(-0.5,-0.125),(0,0),(2,0.25),(4,0.125),(5,0),(6,-0.125),(8,-0.125),(9,0),(10,0.08)]

A = LineString(A_pts)
B = LineString(B_pts)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

ax.plot([a[0] for a in A_pts],[a[1] for a in A_pts],'b')
ax.plot([b[0] for b in B_pts],[b[1] for b in B_pts],'r')
# ax.set_ylim(-1,4)

sys.exit(0)



#%% Create input graph
edgelist = [(0,1),(1,2),(2,3),(3,4),(1,4),(3,5),(2,6)]
node_cord = {0:(0,0), 1:(3,0), 2:(2,np.sqrt(3)), 3:(3,2*np.sqrt(3)),
             4:(5,2*np.sqrt(3)), 5:(4,np.sqrt(8)+np.sqrt(12)), 
             6:(1,np.sqrt(3)+np.sqrt(35))}

G = nx.Graph()
G.add_edges_from(edgelist)

node_geom = {}
edge_geom = {}
edge_length = {}


for n in G.nodes:
    node_geom[n] = Point(node_cord[n])
    
for e in G.edges:
    edge_geom[e] = LineString([node_cord[e[0]],node_cord[e[1]]])
    edge_length[e] = edge_geom[e].length

nx.set_node_attributes(G, node_geom, 'geometry')
nx.set_edge_attributes(G, edge_geom, 'geometry')
nx.set_edge_attributes(G, edge_length, 'length')

#%% Plot graph
d = {'edges':edgelist,
     'geometry':[G.edges[e]['geometry'] for e in edgelist]}
df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
d = {'nodes':G.nodes,
     'geometry':[G.nodes[n]['geometry'] for n in G]}
df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")


#%% Interpolate points along edge geometry
pointlist = []
pointgeom = []
num_pts = 500
count = 100
for e in G.edges:
    geom = G.edges[e]["geometry"]
    length = G.edges[e]["length"]
    for i in range(num_pts+1):
        x,y = geom.interpolate(i/num_pts,normalized=True).xy
        xy = (x[0],y[0])
        count += 1
        
        # Add to the dataframe
        pointlist.append(count)
        pointgeom.append(Point(xy))

d = {'points':pointlist,'geometry':pointgeom}
df_points = gpd.GeoDataFrame(d, crs="EPSG:4326")


#%% Compute distance metric

def distance(geomA,geomB,net):
    distA = {e: net.edges[e]["geometry"].distance(geomA) for e in net.edges}
    edgeA = min(distA, key=distA.get)
    
    distB = {e: net.edges[e]["geometry"].distance(geomB) for e in net.edges}
    edgeB = min(distB, key=distB.get)
    
    if edgeA == edgeB:
        return geomA.distance(geomB)
    else:
        nodeA1,nodeA2 = edgeA
        nodeB1,nodeB2 = edgeB
        
        distA1 = geomA.distance(net.nodes[nodeA1]["geometry"])
        distA2 = geomA.distance(net.nodes[nodeA2]["geometry"])
        distB1 = geomB.distance(net.nodes[nodeB1]["geometry"])
        distB2 = geomB.distance(net.nodes[nodeB2]["geometry"])
        
        dist11 = nx.shortest_path_length(net,nodeA1,nodeB1,weight='length')
        dist12 = nx.shortest_path_length(net,nodeA1,nodeB2,weight='length')
        dist21 = nx.shortest_path_length(net,nodeA2,nodeB1,weight='length')
        dist22 = nx.shortest_path_length(net,nodeA2,nodeB2,weight='length')
        
        d11 = distA1+dist11+distB1
        d12 = distA1+dist12+distB2
        d21 = distA2+dist21+distB1
        d22 = distA2+dist22+distB2
        
        return min(d11,d12,d21,d22)

#%% Compute network distance
s_geom = Point(2,0)
y_pts = []
x_pts = []
for pt_geom in pointgeom:
    x_pts.append(pt_geom.x)
    y_pts.append(distance(pt_geom,s_geom,G))


#%% Plot the persistence diagram
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
df_edges.plot(ax=ax1,edgecolor='red',linewidth=2.0,linestyle='dashed',alpha=1.0)
df_nodes.plot(ax=ax1,color='black',markersize=30,alpha=1.0)
df_points.plot(ax=ax1,color='blue',markersize=1,alpha=1.0)

ax2 = fig.add_subplot(122)
ax2.scatter(x_pts,y_pts,marker='*',s=20,c='red')
ax2.hlines(range(10), 0, 1, transform=ax2.get_yaxis_transform(), 
           colors='blue',linestyle='dotted')

















