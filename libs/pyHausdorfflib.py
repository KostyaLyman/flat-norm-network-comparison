# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:29:36 2022

@author: Rounak Meyur
"""

from pygeodesy import hausdorff_, hypot
from shapely.geometry import Point, LineString
import numpy as np
from geographiclib.geodesic import Geodesic

def geodist(geomA,geomB):
    if type(geomA) != Point: geomA = Point(geomA)
    if type(geomB) != Point: geomB = Point(geomB)
    geod = Geodesic.WGS84
    return geod.Inverse(geomA.y, geomA.x, geomB.y, geomB.x)['s12']

def euclid_dist(geomA,geomB):
    if type(geomA) != Point: geomA = Point(geomA)
    if type(geomB) != Point: geomB = Point(geomB)
    return hypot((geomA.x-geomB.x),(geomA.y-geomB.y))

def interpolate_points(geometry,sep=10):
    points = []
    if not isinstance(geometry, list):
        geometry = [geometry]
    
    # iterate over each linestring geometry
    for geom in geometry:
        new_points = []
        length = geodist(geom.coords[0],geom.coords[1])
        for i in np.arange(0,length,sep):
            x,y = geom.interpolate(i/length,normalized=True).xy
            new_points.append((x[0],y[0]))
        if len(new_points)==0:
            new_points.extend([(geom.coords[0]),(geom.coords[1])])
        
        # Extend the points list
        points.extend(new_points)
    return points

def compute_length(geometry, distance="euclidean"):
    if not isinstance(geometry, list):
        geometry = [geometry]
    if distance == "euclidean":
        length = sum([geom.length for geom in geometry])
    elif distance == "geodesic":
        length = sum([geodist(geom.coords[0],geom.coords[1]) \
                      for geom in geometry])
    return length


def compute_hausdorff(act_geom, synt_geom, distance="euclidean"):
    act_geom_pts = interpolate_points(act_geom)
    synt_geom_pts = interpolate_points(synt_geom)
    # check_hausdorff(act_geom_pts, synt_geom_pts)
    if distance == "euclidean":
        hd1,i1,j1,_,_,_ = hausdorff_(act_geom_pts,synt_geom_pts,
                                  distance=euclid_dist)
        hd2,i2,j2,_,_,_ = hausdorff_(synt_geom_pts,act_geom_pts,
                                  distance=euclid_dist)
    elif distance == "geodesic":
        hd1,i1,j1,_,_,_ = hausdorff_(act_geom_pts,synt_geom_pts,
                                  distance=geodist)
        hd2,i2,j2,_,_,_ = hausdorff_(synt_geom_pts,act_geom_pts,
                                  distance=geodist)
    if hd1 >= hd2:
        hd = hd1
        hd_geom = LineString((Point(act_geom_pts[i1]), 
                              Point(synt_geom_pts[j1])))
    else:
        hd = hd2
        hd_geom = LineString((Point(synt_geom_pts[i2]), 
                              Point(act_geom_pts[j2])))
    return hd, hd_geom
