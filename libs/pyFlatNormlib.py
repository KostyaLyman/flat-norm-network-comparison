# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:00:01 2022

Author: Rounak Meyur

Description: functions to compute the flat norm between currents
"""

from __future__ import absolute_import

import numpy as np
from scipy import sparse
from shapely.geometry import LineString, Point, MultiLineString
import triangle as tr

from libs.pyUtilslib import simpvol, boundary_matrix
from libs.pyLPsolverlib import lp_solver
from libs.pyGeometrylib import geodist

from timeit import default_timer as timer
from datetime import timedelta

def get_geometry(geometry):
    vertices = [Point(c) for c in geometry.coords]
    return [LineString((pt1, pt2)) \
            for pt1, pt2 in zip(vertices, vertices[1:]) \
            if geodist(pt1, pt2) > 1e-6]


def get_current(triangle_structure, geometry) -> np.array:
    current = []
    vertices = triangle_structure['vertices'].tolist()
    for edge in triangle_structure['edges'].tolist():
        vert1 = Point(vertices[edge[0]])
        vert2 = Point(vertices[edge[1]])
        forward_geom = LineString((vert1, vert2))
        reverse_geom = LineString((vert2, vert1))
        for_eq = [forward_geom.equals_exact(geom, 1e-6) for geom in geometry]
        rev_eq = [reverse_geom.equals_exact(geom, 1e-6) for geom in geometry]
        if sum(for_eq) > 0:
            current.append(1)
        elif sum(rev_eq) > 0:
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
        segments.append((ind1, ind2))
    struct = {'vertices': np.array([v.coords[0] for v in vertices]),
              'segments': np.array(segments)}
    return struct


def get_segments(points, start):
    sorted_indices = np.argsort([p.distance(start) for p in points])
    sorted_points = [points[i] for i in sorted_indices]
    return [LineString((sorted_points[i], sorted_points[i + 1])) \
            for i in range(len(points) - 1)]


def update_segment(segments, vertices):
    seg = []
    for g in segments:
        pt_list = []
        for pt in vertices:
            if g.distance(Point(pt)) < 1e-8:
                pt_list.append(Point(pt))
        seg.extend(get_segments(pt_list, Point(g.coords[0])))
    return seg


def prepare_triangulation(segments1, segments2):
    # Add rectangle envelope bounding the segments
    all_lines = MultiLineString(segments1 + segments2)
    rect_env = list(all_lines.minimum_rotated_rectangle.buffer(1e-4).boundary.coords)
    extra_geom = [LineString((Point(rect_env[i]), Point(rect_env[i + 1]))) \
                  for i in range(len(rect_env) - 1)]

    # Structure with added segments
    struct = get_structure(extra_geom + segments1 + segments2)
    return struct


def perform_triangulation(act_geom, syn_geom, adj=1, verbose=False, opts='ps'):
    # Initialize dictionary
    dict_struct = {}

    # Prepare triangulation
    struct = prepare_triangulation(act_geom, syn_geom)
    dict_struct['intermediate'] = struct
    if verbose:
        print("Task completed: Obtained vertices and segments for constrained triangulation")

    # Perform constrained triangulation
    vertices = struct['vertices']
    adj_vertices = adj * (vertices + np.column_stack([[80] * len(vertices),
                                                      [-37] * len(vertices)]))
    adj_struct = {'vertices': adj_vertices,
                  'segments': struct['segments']}
    # try:
    if verbose:
        start_tri = timer()
        print("Task started: Performed triangulation on points")

    # adj_tri_struct = tr.triangulate(adj_struct, opts='psV')
    # adj_tri_struct = tr.triangulate(adj_struct, opts='sV')
    adj_tri_struct = tr.triangulate(adj_struct, opts=opts)
    if verbose:
        end_tri = timer()
        print(f"Task completed: Performed triangulation on points : t={timedelta(seconds=end_tri-start_tri)}")

    adj_tri_vertices = adj_tri_struct['vertices']
    adj_mat = np.column_stack([[80] * len(adj_tri_vertices),
                               [-37] * len(adj_tri_vertices)])
    tri_struct = {'vertices': (adj_tri_vertices / adj) - adj_mat,
                  'segments': adj_tri_struct['segments'],
                  'triangles': adj_tri_struct['triangles']}

    edges = []
    for tgl in tri_struct['triangles']:
        if ([tgl[0], tgl[1]] not in edges) and ([tgl[1], tgl[0]] not in edges):
            edges.append([tgl[0], tgl[1]])
        if ([tgl[1], tgl[2]] not in edges) and ([tgl[2], tgl[1]] not in edges):
            edges.append([tgl[1], tgl[2]])
        if ([tgl[2], tgl[0]] not in edges) and ([tgl[0], tgl[2]] not in edges):
            edges.append([tgl[2], tgl[0]])
    tri_struct['edges'] = np.array(edges)
    dict_struct['triangulated'] = tri_struct
    if verbose:
        print("Task completed: Performed triangulation on points")

    # update input geometries with intersecting points
    V = tri_struct['vertices'].tolist()
    dict_struct['actual'] = update_segment(act_geom, V)
    dict_struct['synthetic'] = update_segment(syn_geom, V)
    if verbose:
        print("Task completed: Updated geometries with intersecting points")

    # except:
    #     print("Failed triangulation!!!")
    #     dict_struct['actual'] = act_geom
    #     dict_struct['synthetic'] = syn_geom
    #     dict_struct['triangulated'] = None

    return dict_struct


# %% flat norm computation
def msfn(points, simplices, subsimplices, input_current, lambda_,
         w=[], v=[], cons=[], k=1, normalized=False):
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
    c = np.concatenate((abs(w), abs(w), lambda_ * abs(v), lambda_ * abs(v)))
    c.reshape(len(c), 1)
    sol, norm = lp_solver(c, cons, input_current)
    x = (sol[0:m_subsimplices] - sol[m_subsimplices:2 * m_subsimplices]).reshape((1, m_subsimplices)).astype(int)
    s = (
            sol[2 * m_subsimplices:2 * m_subsimplices + n_simplices] -
            sol[2 * m_subsimplices + n_simplices:]
    ).reshape(1, n_simplices).astype(int)

    # Compute the two parts of the norm
    norm_subsimplices = np.dot(abs(x), abs(w))[0]
    norm_simplices = np.dot(abs(s), abs(v))[0]

    # Normalize by total length of current
    if normalized:
        input_current_w = np.dot(abs(input_current), abs(w))
        norm = norm / input_current_w

    return x, s, norm, norm_subsimplices, norm_simplices, w
