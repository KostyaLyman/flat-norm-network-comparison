# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:38 2021

Author: Rounak
Description: Functions to create network representations and color graphs based
on their attributes.
"""

from shapely.geometry import Point,LineString,Polygon
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection

#%% Network Geometries
def DrawNodes(synth_graph,ax,label=['S','T','H'],color='green',size=25,
              alpha=1.0):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    # Get the nodes for the specified label
    if label == []:
        nodelist = list(synth_graph.nodes())
    else:
        nodelist = [n for n in synth_graph.nodes() \
                    if synth_graph.nodes[n]['label']==label \
                        or synth_graph.nodes[n]['label'] in label]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha)
    return

def DrawEdges(synth_graph,ax,label=['P','E','S'],color='black',width=2.0,
              style='solid',alpha=1.0):
    """
    """
    # Get the nodes for the specified label
    if label == []:
        edgelist = list(synth_graph.edges())
    else:
        edgelist = [e for e in synth_graph.edges() \
                    if synth_graph[e[0]][e[1]]['label']==label\
                        or synth_graph[e[0]][e[1]]['label'] in label]
    d = {'edges':edgelist,
         'geometry':[synth_graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,linestyle=style,alpha=alpha)
    return

def plot_gdf(ax,df_edges,df_nodes,color):
    """"""
    # df_edges.plot(ax=ax,edgecolor=color,linewidth=1.0)
    df_nodes.plot(ax=ax,color=color,markersize=1)
    return

def plot_network(net,inset={},path=None,with_secnet=False):
    """
    """
    fig = plt.figure(figsize=(40,40), dpi=72)
    ax = fig.add_subplot(111)
    # Draw nodes
    DrawNodes(net,ax,label='S',color='dodgerblue',size=2000)
    DrawNodes(net,ax,label='T',color='green',size=25)
    DrawNodes(net,ax,label='R',color='black',size=2.0)
    if with_secnet: DrawNodes(net,ax,label='H',color='crimson',size=2.0)
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=2.0)
    DrawEdges(net,ax,label='E',color='dodgerblue',width=2.0)
    if with_secnet: DrawEdges(net,ax,label='S',color='crimson',width=1.0)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax,inset[sub]['zoom'],loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes
        DrawNodes(inset[sub]['graph'],axins,label='S',color='dodgerblue',
                  size=2000)
        DrawNodes(inset[sub]['graph'],axins,label='T',color='green',size=25)
        DrawNodes(inset[sub]['graph'],axins,label='R',color='black',size=2.0)
        if with_secnet: DrawNodes(inset[sub]['graph'],axins,label='H',
                                  color='crimson',size=2.0)
        # Draw edges
        DrawEdges(inset[sub]['graph'],axins,label='P',color='black',width=2.0)
        DrawEdges(inset[sub]['graph'],axins,label='E',color='dodgerblue',width=2.0)
        if with_secnet: DrawEdges(inset[sub]['graph'],axins,label='S',
                                  color='crimson',width=1.0)
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], 
                   loc2=inset[sub]['loc2'], fc="none", ec="0.5")
    
    # Legend for the plot
    leghands = [Line2D([0], [0], color='black', markerfacecolor='black', 
                   marker='o',markersize=0,label='primary network'),
            Line2D([0], [0], color='dodgerblue', 
                   markerfacecolor='dodgerblue', marker='o',
                   markersize=0,label='high voltage feeder'),
            Line2D([0], [0], color='white', markerfacecolor='green', 
                   marker='o',markersize=20,label='transformer'),
            Line2D([0], [0], color='white', markerfacecolor='dodgerblue', 
                   marker='o',markersize=20,label='substation')]
    if with_secnet:
        leghands.insert(1,Line2D([0], [0], color='crimson', markerfacecolor='crimson', 
               marker='o',markersize=0,label='secondary network'))
        leghands.insert(-1,Line2D([0], [0], color='white', markerfacecolor='red', 
               marker='o',markersize=20,label='residence'))
    ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 25})
    if path != None: 
        fig.savefig("{}{}.png".format(path,'-51121-dist'),bbox_inches='tight')
    return

def plot_road_network(net,subs,inset={},path=None):
    """
    """
    fig = plt.figure(figsize=(40,40), dpi=72)
    ax = fig.add_subplot(111)
    
    sub_x = [subs[s]['cord'][0] for s in subs]
    sub_y = [subs[s]['cord'][1] for s in subs]
    # Draw nodes
    ax.scatter(sub_x,sub_y,c='dodgerblue',s=2000)
    DrawNodes(net,ax,label='T',color='green',size=25)
    DrawNodes(net,ax,label='R',color='black',size=2.0)
    
    # Draw edges
    d = {'edges':list(net.edges()),
         'geometry':[LineString((net.nodes[e[0]]['cord'],net.nodes[e[1]]['cord'])) \
                     for e in net.edges()]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor="black",linewidth=2.0,linestyle="dashed")
    
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    # Inset figures
    for sub in inset:
        axins = zoomed_inset_axes(ax,inset[sub]['zoom'],loc=inset[sub]['loc'])
        axins.set_aspect(1.3)
        # Draw nodes
        ax.scatter([subs[sub]['cord'][0]],[subs[sub]['cord'][1]],c='dodgerblue',s=2000)
        DrawNodes(inset[sub]['graph'],axins,label='T',color='green',size=25)
        DrawNodes(inset[sub]['graph'],axins,label='R',color='black',size=2.0)
        
        # Draw edges
        d = {'edges':list(inset[sub]['graph'].edges()),
             'geometry':[LineString((inset[sub]['graph'].nodes[e[0]]['cord'],
                                     inset[sub]['graph'].nodes[e[1]]['cord'])) \
                         for e in inset[sub]['graph'].edges()]}
        df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_edges.plot(ax=axins,edgecolor="black",linewidth=2.0,linestyle="dashed")
        
        axins.tick_params(bottom=False,left=False,
                          labelleft=False,labelbottom=False)
        mark_inset(ax, axins, loc1=inset[sub]['loc1'], 
                   loc2=inset[sub]['loc2'], fc="none", ec="0.5")
    
    # Legend for the plot
    leghands = [Line2D([0], [0], color='black', markerfacecolor='black', 
                   marker='o',markersize=0,label='road network'),
            Line2D([0], [0], color='white', markerfacecolor='green', 
                   marker='o',markersize=20,label='transformer'),
            Line2D([0], [0], color='white', markerfacecolor='black', 
                   marker='o',markersize=20,label='road node'),
            Line2D([0], [0], color='white', markerfacecolor='dodgerblue', 
                   marker='o',markersize=20,label='substation')]
    ax.legend(handles=leghands,loc='best',ncol=1,prop={'size': 25})
    if path != None: 
        fig.savefig("{}{}.png".format(path,'-51121-road'),bbox_inches='tight')
    return



#%% Plot the spatial distribution
def get_polygon(boundary):
    """Gets the vertices for the boundary polygon"""
    vert1 = [boundary.west_edge,boundary.north_edge]
    vert2 = [boundary.east_edge,boundary.north_edge]
    vert3 = [boundary.east_edge,boundary.south_edge]
    vert4 = [boundary.west_edge,boundary.south_edge]
    return np.array([vert1,vert2,vert3,vert4])


def plot_deviation(ax,gridlist,C_masked,colormap=cm.BrBG,vmin=-100.0,vmax=100.0):
    x_array = np.array(sorted(list(set([g.west_edge for g in gridlist]\
                                       +[g.east_edge for g in gridlist]))))
    y_array = np.array(sorted(list(set([g.south_edge for g in gridlist]\
                                       +[g.north_edge for g in gridlist]))))
    # Initialize figure
    
    LEFT = min(x_array); RIGHT = max(x_array)
    BOTTOM = min(y_array); TOP = max(y_array)
    ax.set_xlim(LEFT,RIGHT)
    ax.set_ylim(BOTTOM,TOP)
    
    # Plot the grid colors
    ky = len(x_array) - 1
    kx = len(y_array) - 1
    
    ax.pcolor(x_array,y_array,C_masked.reshape((kx,ky)).T,cmap=colormap,
              edgecolor='black',vmin=vmin,vmax=vmax)
    
    # Get the boxes for absent actual data
    verts_invalid = [get_polygon(bound) for i,bound in enumerate(gridlist) \
                    if C_masked.mask[i]]
    c = PolyCollection(verts_invalid,hatch=r"./",facecolor='white',edgecolor='black')
    ax.add_collection(c)
    
    # Plot the accessory stuff
    ax.set_xticks([])
    ax.set_yticks([])
    return

def add_colorbar(fig,ax,vmin=-100.0,vmax=100.0,
                 colormap=cm.BrBG,devname="Percentage Deviation"):
    cobj = cm.ScalarMappable(cmap=colormap)
    cobj.set_clim(vmin=vmin,vmax=vmax)
    cbar = fig.colorbar(cobj,ax=ax)
    cbar.set_label(devname,size=20)
    cbar.ax.tick_params(labelsize=20)
    return


#%%

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

def get_vertseg_geometry(struct):
    if isinstance(struct,list):
        struct = get_structure(struct)
    geom_vertices = [Point(v) for v in struct['vertices'].tolist()]
    geom_segments = [LineString((struct['vertices'][c[0]],
                                 struct['vertices'][c[1]])) \
                      for c in struct['segments']]
    return geom_vertices, geom_segments


def plot_result(dict_struct,t1,t2,s,x):
    # Extract the structures
    tri_struct = dict_struct['triangulated']
    act_geom = dict_struct['actual']
    syn_geom = dict_struct['synthetic']
    struct = dict_struct['intermediate']
    
    # Get triangles and edges from the dictionary
    vertices = tri_struct['vertices']
    triangles = tri_struct['triangles'][s[0]!=0]
    edges = tri_struct['edges'][x[0]!=0]

    geom_triangles = [Polygon(vertices[np.append(t,t[0])]) for t in triangles]
    geom_edges = [LineString(vertices[e]) for e in edges]

    
    geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                     tri_struct['vertices'][c[1]])) \
                          for c in tri_struct['edges']]
    
    # Plot 1: Plot the geometries of the pair of networks
    fig = plt.figure(figsize=(120,60))
    ax1 = fig.add_subplot(221)
    ax1 = plot_input(act_geom,syn_geom,ax1)
    
    # Plot 2: All segments and points in the pre-triangulated phase
    ax2 = fig.add_subplot(222)
    ax2 = plot_intermediate_result(struct,ax2)
    
    # Plot 3: Post-triangulation phase with currents
    ax3 = fig.add_subplot(223)
    ax3 = plot_triangulation(tri_struct,t1,t2,ax3)
    
    # Plot 4: flat norm computated simplices
    ax4 = fig.add_subplot(224)
    draw_points(ax4,geom_vertices,color='black',size=20,alpha=0.5,marker='o')
    draw_lines(ax4,geom_subsimplices,color='black',width=0.5,style='dashed',alpha=0.2,
                directed=False)
    draw_lines(ax4,geom_edges,color='green',width=3.0,style='solid',alpha=1.0,
               directed=False)
    draw_polygons(ax4,geom_triangles,color='magenta',alpha=0.4,label=None)
    ax4.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    return fig

def plot_norm(tri_struct,x,s,ax):
    # Get triangles and edges from the dictionary
    vertices = tri_struct['vertices']
    triangles = tri_struct['triangles'][s[0]!=0]
    edges = tri_struct['edges'][x[0]!=0]

    geom_triangles = [Polygon(vertices[np.append(t,t[0])]) for t in triangles]
    geom_edges = [LineString(vertices[e]) for e in edges]

    
    geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                     tri_struct['vertices'][c[1]])) \
                          for c in tri_struct['edges']]
    
    # Plot 4: flat norm computated simplices
    draw_points(ax,geom_vertices,color='black',size=20,alpha=0.5,marker='o')
    draw_lines(ax,geom_subsimplices,color='black',width=0.5,style='dashed',alpha=0.2,
                directed=False)
    draw_lines(ax,geom_edges,color='green',width=3.0,style='solid',alpha=1.0,
               directed=False)
    draw_polygons(ax,geom_triangles,color='magenta',alpha=0.2,label=None)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    return ax


def plot_intermediate_result(struct,ax):
    # Get the geometries
    geom_all_vertices = [Point(v) for v in struct['vertices'].tolist()]
    geom_all_segments = [LineString((struct['vertices'][c[0]],
                                     struct['vertices'][c[1]])) \
                          for c in struct['segments']]
    
    # Plot 2: All segments and points in the pre-triangulated phase
    draw_points(ax,geom_all_vertices,color='magenta',size=20,alpha=1.0,marker='o')
    draw_lines(ax,geom_all_segments,color='magenta',width=1.0,style='solid',alpha=1.0,
                directed=False)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    return ax


def plot_input(act_geom,syn_geom,ax):
    # Get the geometries
    geom1_vertices,geom1_segments = get_vertseg_geometry(act_geom)
    geom2_vertices,geom2_segments = get_vertseg_geometry(syn_geom)
    # Plot 1: Plot the geometries of the pair of networks
    draw_points(ax,geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
    draw_lines(ax,geom1_segments,color='red',width=2.0,style='solid',alpha=1.0,
               directed=False,label='Actual Network')
    draw_points(ax,geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
    draw_lines(ax,geom2_segments,color='blue',width=2.0,style='solid',alpha=1.0,
               directed=False,label='Synthetic Network')
    # ax.legend(fontsize=20, markerscale=3)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    return ax


def plot_triangulation(tri_struct,t1,t2,ax):
    geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                     tri_struct['vertices'][c[1]])) \
                          for c in tri_struct['edges']]
    
    geom_segment1 = [geom_subsimplices[i] for i,t in enumerate(t1) if t!=0]
    geom_segment2 = [geom_subsimplices[i] for i,t in enumerate(t2) if t!=0]
    
    draw_points(ax,geom_vertices,color='black',size=20,alpha=0.5,marker='o')
    draw_lines(ax,geom_subsimplices,color='black',width=0.5,style='dashed',
               alpha=0.2, directed=False)
    draw_lines(ax,geom_segment1,color='red',width=2.0,style='solid',
               alpha=1.0, directed=False)
    draw_lines(ax,geom_segment2,color='blue',width=2.0,style='solid',
               alpha=1.0, directed=False)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    return ax


def plot_regions(act_geom,syn_geom,geom_regions,ax):
    # Get the geometries
    geom1_vertices,geom1_segments = get_vertseg_geometry(act_geom)
    geom2_vertices,geom2_segments = get_vertseg_geometry(syn_geom)
    # Plot 1: Plot the geometries of the pair of networks
    draw_points(ax,geom1_vertices,color='red',size=20,alpha=1.0,marker='o')
    draw_lines(ax,geom1_segments,color='red',width=2.0,style='solid',alpha=1.0,
               directed=False,label='Actual Network')
    draw_points(ax,geom2_vertices,color='blue',size=20,alpha=1.0,marker='o')
    draw_lines(ax,geom2_segments,color='blue',width=2.0,style='solid',alpha=1.0,
               directed=False,label='Synthetic Network')
    # ax.legend(fontsize=20, markerscale=3)
    draw_polygons(ax,geom_regions,color='cyan',alpha=0.2,label=None)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    return ax


def plot_failed_triangulation(dict_struct):
    act_geom = dict_struct['actual']
    syn_geom = dict_struct['synthetic']
    struct = dict_struct['intermediate']
    # Plot 1: Plot the geometries of the pair of networks
    fig = plt.figure(figsize=(120,30))
    ax1 = fig.add_subplot(121)
    ax1 = plot_input(act_geom,syn_geom,ax1)
    
    # Plot 2: All segments and points in the pre-triangulated phase
    ax2 = fig.add_subplot(122)
    ax2 = plot_intermediate_result(struct,ax2)
    return fig







