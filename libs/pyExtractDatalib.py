# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:46:36 2018

Authors: Rounak Meyur

Description: This library contains classes, methods etc to extract data from
the noldor database with required credentials. 
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString,Point
from collections import namedtuple as nt
from pyGeometrylib import Link





def GetHomes(path,fis):
    '''
    '''
    df_home = pd.read_csv(path+'load/'+fis+'-home-load.csv')
    df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:27].values,axis=1))
    df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:27].values,axis=1))
    
    home = nt("home",field_names=["cord","profile","peak","average"])
    dict_load = df_home.iloc[:,[0]+list(range(3,27))].set_index('hid').T.to_dict('list')
    dict_cord = df_home.iloc[:,0:3].set_index('hid').T.to_dict('list')
    dict_peak = dict(zip(df_home.hid,df_home.peak))
    dict_avg = dict(zip(df_home.hid,df_home.average))
    homes = home(cord=dict_cord,profile=dict_load,peak=dict_peak,average=dict_avg)
    return homes


#%% Get output synthetic networks (output of Step 2)
def GetDistNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'H' for home, 'T' for transformer, 'R' for road node, 
                    'S' for subs
            voltage: node voltage in pu
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
            geometry: shapely geometry of edge
            geo_length: length of edge in meters
            flow: power flowing in kVA through edge
    """
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(path+str(c)+'-dist-net.gpickle')
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(path+str(code)+'-dist-net.gpickle')
    return graph



#%% Actual Network and Synthetic Network data
def get_geometry(path,area):
    """
    Loads the shape files for edges and nodes in the region and returns the 
    geometry of the network.

    Parameters
    ----------
    path : string
        path of directory where the shape files are stored.
    area : dictionary
        data keyed by area ID and value as the root node.

    Returns
    -------
    df_buses: geopandas dataframe of nodes
    df_lines: geopandas dataframe of edges
    act_graph: networkx graph of actual network
    limits: tuple of limits
    """
    # Get dataframe for buses and lines
    df_lines = gpd.read_file(path+area+'/'+area+'_edges.shp')
    df_buses = gpd.read_file(path+area+'/'+area+'_nodes.shp')
    
    # Get dictionary of node geometries and edgelist
    edgelist = [tuple([int(x) for x in e.split('_')]) for e in df_lines['ID']]
    
    # Create actual graph network
    act_graph = nx.Graph()
    act_graph.add_edges_from(edgelist)
    
    # Compute edge lengths
    geo_length = {}
    for e in edgelist:
        edge = str(e[0])+'_'+str(e[1])
        edge_geom = df_lines.loc[df_lines.ID == edge]['geometry'].values[0]
        geo_length[e] = Link(edge_geom).geod_length
    nx.set_edge_attributes(act_graph,geo_length,'geo_length')

    # Get coordinate limits
    busgeom = df_buses['geometry']
    x_bus = [geom.x for geom in busgeom]
    y_bus = [geom.y for geom in busgeom]
    
    # Get axes limits
    buffer = 0.0
    left = min(x_bus)-buffer
    right = max(x_bus)+buffer
    bottom = min(y_bus)-buffer
    top = max(y_bus)+buffer
    limits = (left,right,bottom,top)
    
    return df_buses,df_lines,act_graph,limits


def get_network(synth,limits):
    """
    Gets the actual network, extracts the limits of the region and loads the
    synthetic network of the region.

    Parameters
    ----------
    synth : networkx graph
        total synthetic primary network
    limits: tuple of float data
        axes limits for the area under consideration

    Returns
    -------
    act_graph: networkx graph of the actual network
    synth_graph: networkx graph of the synthetic network in the axes limits
    limits: axes limits for the area under consideration

    """
    # Get the primary network in the region from the synthetic network
    left,right,bottom,top = limits
    nodelist = [n for n in synth.nodes \
                if left<=synth.nodes[n]['cord'][0]<=right \
                 and bottom<=synth.nodes[n]['cord'][1]<=top \
                     and synth.nodes[n]['label'] != 'H']
    synth_graph = nx.subgraph(synth,nodelist)
    
    # Get the dataframe for node and edge geometries
    d = {'nodes':list(synth_graph.nodes()),
         'geometry':[Point(synth_graph.nodes[n]['cord']) \
                     for n in synth_graph.nodes()]}
    df_cords = gpd.GeoDataFrame(d, crs="EPSG:4326")
    
    d = {'edges':list(synth_graph.edges()),
         'geometry':[synth_graph[e[0]][e[1]]['geometry'] \
                     for e in synth_graph.edges()]}
    df_synth = gpd.GeoDataFrame(d, crs="EPSG:4326")
    return synth_graph,df_cords,df_synth

def get_areadata(path,area,root,synth):
    """
    Extracts the data for an area and returns them in a dictionary which
    is labeled with the required entries.

    Parameters
    ----------
    path : string type
        directory path for the shape files of the actual network
    area : string type
        ID of the area for the shape file
    root : integer type
        root node for the area in the shape file
    synth : networkx graph
        complete synthetic primary distribution network

    Returns
    -------
    data: dictionary of area data
        root: root node ID of actual network
        df_lines: geopandas data frame of actual network edges
        df_buses: geopandas data frame of actual network nodes
        actual: networkx graph of actual network
        synthetic: networkx graph of synthetic network
        limits: axes limits of the region of comparison
        df_synth: geopandas data frame of synthetic network edges
        df_cords: geopandas data frame of synthetic network nodes

    """
    df_buses,df_lines,act_graph,limits = get_geometry(path,area)
    synth_graph,df_cords,df_synth = get_network(synth,limits)
    # Store the data in the dictionary
    data = {'root':root,'limits':limits,'df_lines':df_lines,
            'df_buses':df_buses,'actual':act_graph,
            'synthetic':synth_graph,'df_synth':df_synth,'df_cords':df_cords}
    return data






















