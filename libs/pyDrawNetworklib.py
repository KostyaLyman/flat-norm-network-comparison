# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:38 2021

Author: Rounak
Description: Functions to create network representations and color graphs based
on their attributes.
"""

from shapely.geometry import Point, LineString, Polygon
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# %% Plot the spatial distribution
def get_polygon(boundary):
    """Gets the vertices for the boundary polygon"""
    vert1 = [boundary.west_edge, boundary.north_edge]
    vert2 = [boundary.east_edge, boundary.north_edge]
    vert3 = [boundary.east_edge, boundary.south_edge]
    vert4 = [boundary.west_edge, boundary.south_edge]
    return np.array([vert1, vert2, vert3, vert4])


def plot_deviation(ax, gridlist, C_masked, colormap=cm.BrBG, vmin=-100.0, vmax=100.0):
    x_array = np.array(sorted(list(set([g.west_edge for g in gridlist] \
                                       + [g.east_edge for g in gridlist]))))
    y_array = np.array(sorted(list(set([g.south_edge for g in gridlist] \
                                       + [g.north_edge for g in gridlist]))))
    # Initialize figure

    LEFT = min(x_array);
    RIGHT = max(x_array)
    BOTTOM = min(y_array);
    TOP = max(y_array)
    ax.set_xlim(LEFT, RIGHT)
    ax.set_ylim(BOTTOM, TOP)

    # Plot the grid colors
    ky = len(x_array) - 1
    kx = len(y_array) - 1

    ax.pcolor(x_array, y_array, C_masked.reshape((kx, ky)).T, cmap=colormap,
              edgecolor='black', vmin=vmin, vmax=vmax)

    # Get the boxes for absent actual data
    verts_invalid = [get_polygon(bound) for i, bound in enumerate(gridlist) \
                     if C_masked.mask[i]]
    c = PolyCollection(verts_invalid, hatch=r"./", facecolor='white', edgecolor='black')
    ax.add_collection(c)

    # Plot the accessory stuff
    ax.set_xticks([])
    ax.set_yticks([])
    return


def add_colorbar(fig, ax, vmin=-100.0, vmax=100.0,
                 colormap=cm.BrBG, devname="Percentage Deviation"):
    cobj = cm.ScalarMappable(cmap=colormap)
    cobj.set_clim(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cobj, ax=ax)
    cbar.set_label(devname, size=20)
    cbar.ax.tick_params(labelsize=20)
    return


# %%

def draw_points(ax, points, color='red', size=10, alpha=1.0, marker='o', label=None):
    if len(points) == 0:
        return ax
    if isinstance(points, list):
        d = {'nodes': range(len(points)),
             'geometry': [pt_geom for pt_geom in points]}
    elif isinstance(points, dict):
        d = {'nodes': range(len(points)),
             'geometry': [points[k] for k in points]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax, color=color, markersize=size, alpha=alpha, marker=marker, label=label)
    return ax


def draw_lines(ax, lines, color='red', width=2.0, style='solid', alpha=1.0, label=None,
               directed=False):
    if isinstance(lines, LineString):
        lines = [lines]
    if len(lines) == 0:
        return ax
    d = {'edges': range(len(lines)),
         'geometry': [line_geom for line_geom in lines]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax, edgecolor=color, linewidth=width,
                  linestyle=style, alpha=alpha, label=label)
    if directed:
        for line_geom in lines:
            arrow_width = 0.03
            head_width = 2.5 * arrow_width
            head_length = 2 * arrow_width

            cp0 = np.array(line_geom.coords[0])
            cp1 = np.mean(line_geom.coords, axis=0)

            delta = cos, sin = (cp1 - cp0) / np.hypot(*(cp1 - cp0))
            length = Point(cp0).distance(Point(cp1))
            x_pos, y_pos = (
                    (cp0 + cp1) / 2 - delta * length / 2)
            ax.arrow(x_pos, y_pos, cos * length, sin * length, ec=color, fc=color,
                     head_width=head_width, head_length=head_length, ls='--',
                     shape='full', length_includes_head=True)
    return ax


def draw_polygons(ax, polygons, color='red', alpha=1.0, label=None):
    if len(polygons) == 0:
        return ax
    if isinstance(polygons, list):
        d = {'nodes': range(len(polygons)),
             'geometry': [geom for geom in polygons]}
    elif isinstance(polygons, dict):
        d = {'nodes': range(len(polygons)),
             'geometry': [polygons[k] for k in polygons]}
    df_polygons = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_polygons.plot(ax=ax, facecolor=color, alpha=alpha, label=label)
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
        segments.append((ind1, ind2))
    struct = {'vertices': np.array([v.coords[0] for v in vertices]),
              'segments': np.array(segments)}
    return struct


def get_vertseg_geometry(struct):
    if isinstance(struct, list):
        struct = get_structure(struct)
    geom_vertices = [Point(v) for v in struct['vertices'].tolist()]
    geom_segments = [LineString((struct['vertices'][c[0]],
                                 struct['vertices'][c[1]])) \
                     for c in struct['segments']]
    return geom_vertices, geom_segments


def plot_result(dict_struct, t1, t2, s, x):
    # Extract the structures
    tri_struct = dict_struct['triangulated']
    act_geom = dict_struct['actual']
    syn_geom = dict_struct['synthetic']
    struct = dict_struct['intermediate']

    # Get triangles and edges from the dictionary
    vertices = tri_struct['vertices']
    triangles = tri_struct['triangles'][s[0] != 0]
    edges = tri_struct['edges'][x[0] != 0]

    geom_triangles = [Polygon(vertices[np.append(t, t[0])]) for t in triangles]
    geom_edges = [LineString(vertices[e]) for e in edges]

    geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                     tri_struct['vertices'][c[1]])) \
                         for c in tri_struct['edges']]

    # Plot 1: Plot the geometries of the pair of networks
    fig = plt.figure(figsize=(120, 60))
    ax1 = fig.add_subplot(221)
    ax1 = plot_input(act_geom, syn_geom, ax1)

    # Plot 2: All segments and points in the pre-triangulated phase
    ax2 = fig.add_subplot(222)
    ax2 = plot_intermediate_result(struct, ax2)

    # Plot 3: Post-triangulation phase with currents
    ax3 = fig.add_subplot(223)
    ax3 = plot_triangulation(tri_struct, t1, t2, ax3)

    # Plot 4: flat norm computated simplices
    ax4 = fig.add_subplot(224)
    draw_points(ax4, geom_vertices, color='black', size=20, alpha=0.5, marker='o')
    draw_lines(ax4, geom_subsimplices, color='black', width=0.5, style='dashed', alpha=0.2,
               directed=False)
    draw_lines(ax4, geom_edges, color='green', width=3.0, style='solid', alpha=1.0,
               directed=False)
    draw_polygons(ax4, geom_triangles, color='magenta', alpha=0.4, label=None)
    ax4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return fig


def plot_norm(tri_struct, x, s, ax):
    # Get triangles and edges from the dictionary
    vertices = tri_struct['vertices']
    triangles = tri_struct['triangles'][s[0] != 0]
    edges = tri_struct['edges'][x[0] != 0]

    geom_triangles = [Polygon(vertices[np.append(t, t[0])]) for t in triangles]
    geom_edges = [LineString(vertices[e]) for e in edges]

    geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                     tri_struct['vertices'][c[1]])) \
                         for c in tri_struct['edges']]

    # Plot 4: flat norm computated simplices
    draw_points(ax, geom_vertices, color='black', size=20, alpha=0.5, marker='o')
    draw_lines(ax, geom_subsimplices, color='black', width=0.5, style='dashed', alpha=0.2,
               directed=False)
    draw_lines(ax, geom_edges, color='green', width=3.0, style='solid', alpha=1.0,
               directed=False)
    draw_polygons(ax, geom_triangles, color='magenta', alpha=0.2, label=None)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_intermediate_result(struct, ax):
    # Get the geometries
    geom_all_vertices = [Point(v) for v in struct['vertices'].tolist()]
    geom_all_segments = [LineString((struct['vertices'][c[0]],
                                     struct['vertices'][c[1]])) \
                         for c in struct['segments']]

    # Plot 2: All segments and points in the pre-triangulated phase
    draw_points(ax, geom_all_vertices, color='magenta', size=20, alpha=1.0, marker='o')
    draw_lines(ax, geom_all_segments, color='magenta', width=1.0, style='solid', alpha=1.0,
               directed=False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_input(act_geom, syn_geom, ax):
    # Get the geometries
    geom1_vertices, geom1_segments = get_vertseg_geometry(act_geom)
    geom2_vertices, geom2_segments = get_vertseg_geometry(syn_geom)
    # Plot 1: Plot the geometries of the pair of networks
    draw_points(ax, geom1_vertices, color='red', size=20, alpha=1.0, marker='o')
    draw_lines(ax, geom1_segments, color='red', width=2.0, style='solid', alpha=1.0,
               directed=False, label='Actual Network')
    draw_points(ax, geom2_vertices, color='blue', size=20, alpha=1.0, marker='o')
    draw_lines(ax, geom2_segments, color='blue', width=2.0, style='solid', alpha=1.0,
               directed=False, label='Synthetic Network')
    # ax.legend(fontsize=20, markerscale=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_triangulation(tri_struct, t1, t2, ax):
    geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                     tri_struct['vertices'][c[1]])) \
                         for c in tri_struct['edges']]

    geom_segment1 = [geom_subsimplices[i] for i, t in enumerate(t1) if t != 0]
    geom_segment2 = [geom_subsimplices[i] for i, t in enumerate(t2) if t != 0]

    draw_points(ax, geom_vertices, color='black', size=20, alpha=0.5, marker='o')
    draw_lines(ax, geom_subsimplices, color='black', width=0.5, style='dashed',
               alpha=0.2, directed=False)
    draw_lines(ax, geom_segment1, color='red', width=2.0, style='solid',
               alpha=1.0, directed=False)
    draw_lines(ax, geom_segment2, color='blue', width=2.0, style='solid',
               alpha=1.0, directed=False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_regions(act_geom, syn_geom, geom_regions, ax,  **kwargs):
    highlight = kwargs.get('region_highlight', None)
    highlight = kwargs.get('highlight', highlight)
    alpha = kwargs.get('region_alpha', 0.2)
    plot_colors = dict(
        region=kwargs.get('region_color', 'cyan'),
        highlight=kwargs.get('highlight_color', 'orange')
    )

    # Get the geometries
    geom1_vertices, geom1_segments = get_vertseg_geometry(act_geom)
    geom2_vertices, geom2_segments = get_vertseg_geometry(syn_geom)
    # Plot 1: Plot the geometries of the pair of networks
    draw_points(ax, geom1_vertices, color='red', size=20, alpha=1.0, marker='o')
    draw_lines(ax, geom1_segments, color='red', width=2.0, style='solid', alpha=1.0,
               directed=False, label='Actual Network')
    draw_points(ax, geom2_vertices, color='blue', size=20, alpha=1.0, marker='o')
    draw_lines(ax, geom2_segments, color='blue', width=2.0, style='solid', alpha=1.0,
               directed=False, label='Synthetic Network')
    # ax.legend(fontsize=20, markerscale=3)
    if highlight is not None:
        geom_regions = geom_regions[:]
        region = geom_regions.pop(highlight)
        draw_polygons(
            ax, [region],
            color=plot_colors['highlight'], alpha=alpha,
            label=None
        )

    draw_polygons(ax, geom_regions, color=plot_colors['region'], alpha=alpha, label=None)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_failed_triangulation(dict_struct):
    act_geom = dict_struct['actual']
    syn_geom = dict_struct['synthetic']
    struct = dict_struct['intermediate']
    # Plot 1: Plot the geometries of the pair of networks
    fig = plt.figure(figsize=(120, 30))
    ax1 = fig.add_subplot(121)
    ax1 = plot_input(act_geom, syn_geom, ax1)

    # Plot 2: All segments and points in the pre-triangulated phase
    ax2 = fig.add_subplot(122)
    ax2 = plot_intermediate_result(struct, ax2)
    return fig


# %% Single geometry
def plot_demo_input(geom, ax):
    geom_vertices, geom_segments = get_vertseg_geometry(geom)
    draw_points(ax, geom_vertices, color='blue', size=20, alpha=1.0, marker='o')
    draw_lines(ax, geom_segments, color='blue', width=2.0, style='solid', alpha=1.0)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_demo_triangulation(tri_struct, t, ax):
    geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
                                     tri_struct['vertices'][c[1]])) \
                         for c in tri_struct['edges']]

    geom_segment = [geom_subsimplices[i] for i, t_ in enumerate(t) if t_ != 0]

    draw_points(ax, geom_vertices, color='black', size=20, alpha=0.5, marker='o')
    draw_lines(ax, geom_subsimplices, color='black', width=0.7, style='dashed',
               alpha=0.2, directed=False)
    draw_lines(ax, geom_segment, color='blue', width=2.0, style='solid',
               alpha=1.0, directed=False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_demo_norm(tri_struct, x, s, ax):
    # Get triangles and edges from the dictionary
    vertices = tri_struct['vertices']
    triangles = tri_struct['triangles'][s[0] != 0]
    edges = tri_struct['edges'][x[0] != 0]

    geom_triangles = [Polygon(vertices[np.append(t, t[0])]) for t in triangles]
    geom_edges = [LineString(vertices[e]) for e in edges]

    # geom_vertices = [Point(v) for v in tri_struct['vertices'].tolist()]
    # geom_subsimplices = [LineString((tri_struct['vertices'][c[0]],
    #                                  tri_struct['vertices'][c[1]])) \
    #                       for c in tri_struct['edges']]

    # Plot 4: flat norm computated simplices
    draw_lines(ax, geom_edges, color='green', width=3.0, style='solid', alpha=1.0,
               directed=False)
    draw_polygons(ax, geom_triangles, color='magenta', alpha=0.2, label=None)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax


def plot_demo_flatnorm(geom, tri_struct, x, s, ax, offset=0.0):
    geom_offset = geom.parallel_offset(offset, 'left')
    draw_lines(ax, geom_offset, color='blue', width=2.0, style='solid', alpha=1.0)

    # Get triangles and edges from the dictionary
    vertices = tri_struct['vertices']
    triangles = tri_struct['triangles'][s[0] != 0]
    edges = tri_struct['edges'][x[0] != 0]

    geom_triangles = [Polygon(vertices[np.append(t, t[0])]) for t in triangles]
    geom_edges = [LineString(vertices[e]) for e in edges]

    draw_lines(ax, geom_edges, color='green', width=3.0, style='solid', alpha=1.0,
               directed=False)
    draw_polygons(ax, geom_triangles, color='magenta', alpha=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    labels = [r"Input current $T$",
              r"Flat norm $T-\partial S$",
              r"Surface area $S$"]
    handles = [Line2D([0], [0], color='blue', linewidth=3.0),
               Line2D([0], [0], color='green', linewidth=3.0),
               Patch(facecolor='magenta', alpha=0.5)]
    ax.legend(handles, labels, markerscale=4, fontsize=20)
    return ax
