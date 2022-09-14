# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:19:33 2022

@author: Rounak Meyur
"""

from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd


def update_segment(segments,vertices):
    seg = []
    for g in segments:
        # get list of points on the geometry
        pt_list = []
        for pt in vertices:
            if g.distance(Point(pt)) < 1e-8:
                pt_list.append(Point(pt))
        
        # Sort the points according to distance from first end
        start = Point(g.coords[0])
        sorted_indices = np.argsort([p.distance(start) for p in pt_list])
        sorted_points = [pt_list[i] for i in sorted_indices]
        
        # Construct subsegments from adjacent points
        sub_geom = [LineString((sorted_points[i],sorted_points[i+1])) \
                    for i in range(len(pt_list)-1)]
        seg.extend(sub_geom)
    return seg


path = "C:\\Users\\rouna\\Documents\\GitHub\\persistent-homology\\out\\check\\"
df_act_geom = pd.read_csv(path+"input-act-geometry.txt", sep="\t", header=None,
                          names=("start", "end"))

act_geom = []
for i in range(len(df_act_geom)):
    pt1 = Point([float(y) for y in df_act_geom["start"][i].lstrip('(').rstrip(')').split(',')])
    pt2 = Point([float(y) for y in df_act_geom["end"][i].lstrip('(').rstrip(')').split(',')])
    act_geom.append(LineString((pt1,pt2)))

df_vertices = pd.read_csv(path+"output-vertices.txt", sep="\t", header=None,
                          names=("x", "y"))
verts = df_vertices.to_numpy().tolist()


updated_act_geom = update_segment(act_geom,verts)
