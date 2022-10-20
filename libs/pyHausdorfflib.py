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


def compute_hausdorff(act_geom, synt_geom):
    act_geom_pts = interpolate_points(act_geom)
    synt_geom_pts = interpolate_points(synt_geom)
    
    hd,i,j,_,_,_ = hausdorff_(act_geom_pts,synt_geom_pts,distance=euclid_dist)
    return hd, LineString((Point(act_geom_pts[i]), Point(synt_geom_pts[j])))