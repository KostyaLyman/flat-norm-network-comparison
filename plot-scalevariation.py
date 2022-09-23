# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:28:02 2022

Author: Rounak Meyur

Description: Plots the variation in scaled simplicial flat norm with the change
in scale.
"""

from __future__ import absolute_import

import sys,os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx


workpath = os.getcwd()
libpath = workpath + "/libs/"
sys.path.append(libpath)

rootpath = os.path.dirname(workpath)
outpath = workpath + "/out/"

R = 6378
import pandas as pd
df = pd.read_csv(outpath+"flat-norm.txt",sep='\t',
                 header=None,names=["lambda_","norm","norm1","norm2"])

fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(1,1,1)

ax.plot(df.lambda_,df.norm*R,
        color='black',marker='D',
        markersize=3,linestyle='solid')

ax.plot(df.lambda_,df.norm1*R,
        color='red',marker='^',
        markersize=3,linestyle='solid')

ax.set_xlabel("Scale for flat norm",fontsize=20)
ax.set_ylabel("Scaled simplicial flat norm (km)",fontsize=20)

ax_2 = ax.twinx()
ax_2.plot(df.lambda_,df.norm2*R*R,
        color='blue',marker='o',
        markersize=3,linestyle='solid')

ax_2.set_ylabel("Simplicial flat norm (sq.km.)",fontsize=20)

labels = ["Multi-scale simplicial flat norm","Length component",
          "Surface area component"]
colors = ["black","red","blue"]
markers = ["D","^","o"]
han = [Line2D([0], [0], color=color, markerfacecolor=color, 
               marker=mark,markersize=10,label=label) \
              for label, color, mark in zip(labels, colors, markers)]

ax.legend(handles=han,ncol=1,prop={'size': 15},loc='upper center')
fig.savefig(workpath+'/figs/scale-variation.png',bbox_inches='tight')