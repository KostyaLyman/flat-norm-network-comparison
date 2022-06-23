# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:17:48 2022

Author: Rounak Meyur
"""

from numpy import inf,sqrt
import numpy as np

#%% Class: Simplex
class Simplex:
    """
    Simplex: Point, edge, triangle, tetrahedron and higher order simplices
        Attributes:
            vertices: list of vertices forming the simplex
            dim: dimension of the simplex
    """
    def __init__(self,l):
        """
        Initialize the simplex identifier

        Parameters
        ----------
        l : list
            list of vertices which constructs the simplex.

        Returns
        -------
        None.

        """
        self.vertices = l[:]
        self.vertices.sort()
        self.dim = len(l)
        return

    # Method for equating two Simplex objects
    def __eq__(self,other):
        return (self.vertices == other.vertices)
    
    # Method to compare two Simplex objects
    def __lt__(self,other): # should only be used for simplices of same dimension
        for (x,y) in zip(self.vertices,other.vertices):
            if x>y:
                return False
        return True


    def __hash__(self):
        return hash(tuple(self.vertices))
    
    # Inherent method for representing class object in print statement
    def __str__(self):
        return str(self.vertices)
    
    # Inherent method for representing class object in the console
    def __repr__(self):
        return "Simplex{}".format(str(self.vertices))


    def faces(self):
        """
        Returns the (d-1)-dimensional faces of the d-dimensional simplex

        Returns
        -------
        res : list
            list of (d-1) dimensional simplex objects forming the faces of the 
            current simplex object.

        """
        res = []
        for i in range(self.dim):
            res.append(Simplex(self.vertices[:i]+self.vertices[i+1:]))
        return res
    
    
    def isFace(self,other):
        """
        Check if current Simplex object is a face of the other Simplex object

        Parameters
        ----------
        other : Simplex object
            Higher dimensional simplex object to check for face.

        Returns
        -------
        bool
            True, if current Simplex object is a face of the other 
            Simplex object.

        """
        if self.dim+1 != other.dim:
            return False
        for i in self.vertices:
            if i not in other.vertices:
                return False
        return True

#%% Class: SimplexChain

class SimplexChain:
    def __init__(self,simplexCoeffList,scomplex):
        self.coeffs = {}
        self.complex = scomplex
        for (j,c) in simplexCoeffList:
            self.coeffs[j] = c % self.complex.field

    def getCoeff(self,j):
        if j in self.coeffs:
            return self.coeffs[j]
        else:
            return 0

    def purge(self):   # removes simplices with coefficient 0
        toBeRemoved= []
        for j in self.coeffs:
            if self.coeffs[j] == 0:
                toBeRemoved.append(j)
        for j in toBeRemoved:
            self.coeffs.pop(j)

    def isEmpty(self):
        for j in self.coeffs:
            if self.coeffs[j] != 0:
                return False
        return True

    def __add__(self,other):
        res = SimplexChain([],self.complex)
        for j in self.coeffs:
            res.coeffs[j] = self.coeffs[j]
        for j in other.coeffs:
            if j in res.coeffs:
                res.coeffs[j] = (other.coeffs[j] + res.coeffs[j])%self.complex.field
            else:
                res.coeffs[j] = other.coeffs[j]
            if res.coeffs[j] == 0:
                res.coeffs.pop(j)
        return res

    def __neg__(self):
        res = SimplexChain([],self.complex)
        for j in self.coeffs:
            res.coeffs[j] = (-self.coeffs[j])%self.complex.field
        return res


    def __sub__(self,other):
        return self + (-other)

    def __rmul__(self,other):
        res = SimplexChain([],self.complex)
        for j in self.coeffs:
            res.coeffs[j] = (other*self.coeffs[j])%self.complex.field
        return res

    def __str__(self):
        res = []
        for j in self.coeffs:
            res.append(str(self.coeffs[j]) + " * " + str(self.complex.simplices[j]))
        return "  " + "\n+ ".join(res)

    def __repr__(self):
        return str(self)


#%% Functions


def boundary(s,schain):
    res = SimplexChain([],schain.complex)

    for j in schain.coeffs:
        faces = j.faces()
        l = [(faces[i],(-1)**i) for i in range(schain.complex.simplices[j].dim) ]
        res += schain.coeffs[s] * SimplexChain(l,schain.complex)
    return res

def simplexBoundary(s,scomplex):
    res = SimplexChain([],scomplex)
    if s.dim == 1:
        return res
    faces = s.faces()
    l = [(scomplex._indexBySimplex[faces[i]],(-1)**i) for i in range(s.dim) ]
    res += SimplexChain(l,scomplex)
    return res

def euclidianDistance(x,y):
    res = 0
    for i in range(len(x)):
        res += (x[i]-y[i])**2
    return sqrt(res)


#%% RIPs complex
class RipsComplex:
    """
    Class used to process points in a metric space.
    Parameters:
    pointList : List of points.
    distance : function with signature point * point -> float
    threshold : float, maximum distance which will be considered
    when constructing the complex
    verbose: bool, set to True for info on computation progress
    """
    def __init__(self, pointList, distance = euclidianDistance, 
                 threshold = None,verbose = False):
        self._verbose = verbose
        self.points = pointList[:]
        self.distance = distance
        self.nPoints = len(pointList)
        self.compute_dist_matrix()
        if not threshold:
            self.threshold = max([max(a) for a in self.matrix])+1
        else:
            self.threshold = threshold

    def compute_dist_matrix(self):
        self.matrix = [[ self.distance(x,y) for x in self.points ] for y in self.points]



    def plot(self,threshold = None):
        """
        Plots the 1-skeleton of the points.
        It is possible to specify a threshold
        different from the threshold chosen
        during initialization.
        """
        if not threshold:
            threshold = self.threshold

        for x in self.points:
            plt.plot(x[0],x[1],'ro')
            for y in self.points:
                if x < y and self.distance(x,y) < threshold:
                    plt.plot([x[0],y[0]],[x[1],y[1]],'k-')
        plt.show()


    def compute_skeleton(self,maxDimension = None):
        """
        Computes the Rips-Vietoris complex of the points, with the
        chosen threshold, and up to a given maximum dimension. If
        no maximum dimension is specified, all simplices are computed.
        Algorithm comes from Afra Zomorodian :
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.210.426&rep=rep1&type=pdf
        """
        if not maxDimension:
            maxDimension = self.nPoints


        prevSimplices = []
        currSimplices = []


        self.complex = FilteredComplex()

        # initialize 1-skeleton
        for x in range(self.nPoints):
            for y in range(self.nPoints):
                self.complex.insert([x],0)
                if x < y and self.matrix[x][y] < self.threshold:
                    self.complex.insert([x,y],self.matrix[x][y])
                    currSimplices.append([x,y])


        # compute i+1-skeleton from i-skeleton
        simplexList = []
        for i in range(1,maxDimension):
            if self._verbose:
                print("Computing {}-simplices.".format(i))
            prevSimplices = currSimplices[:]
            currSimplices = []
            count = 0
            total = len(prevSimplices)
            for l in prevSimplices:
                count +=1
                if count%1000 ==0 and self._verbose:
                    print("Step {}: {}/{}".format(i,count,total))
                neighbours = self.lowerNeighbours(l)

                for v in neighbours:
                    s = Simplex(l+[v])
                    simplexList.append(s)
                    currSimplices.append(l+[v])
        if self._verbose:
            print("Done creating skeleton. Computing weights...")

        # Finally, compute the weight of each additional simplex
        total = len(simplexList)
        count = 0
        for s in simplexList:
            count +=1
            if count%1000 ==0 and self._verbose:
                print("{}/{}".format(count,total))
            value = self.computeWeight(s)
            self.complex.append(s,value)


    def lowerNeighbours(self,l):
        res = []
        for v in range(min(l)):
            inNbrs = True
            for u in l:
                if self.matrix[u][v] >= self.threshold:
                    inNbrs = False
                    break
            if inNbrs:
                res.append(v)

        return res

    def computeWeight(self,s):
        res = self.complex.degree(s)
        if res>=0:
            return res

        for f in s.faces():
            res = max(res,self.computeWeight(f))
        return res

#%% Filtered Complex


class FilteredComplex:
    # the degree of a simplex is the lowest index for which it appears in the complex

    def __init__(self,warnings = False):
        self._simplices = [] # list of simplices
        self._degrees_dict = {} # contains the degrees. keys are simplices
        self._numSimplices = 0
        self._dimension = 0
        self._warnings = warnings
        self._maxDeg = 0


    def degree(self,s): # check if simplex s is already in the complex, returns the degree if it is, -1 otherwise
        if s in self._degrees_dict:
            return self._degrees_dict[s]
        else:
            return -1


    def append(self,s,d): 
        #simplex as a list of vertices, degree. Insert so that the order is preserved
        # if the simplex is already in the complex, do nothing
        update = False
        if self.degree(s)>=0:
            if self._warnings:
                print("Face {} is already in the complex.".format(s))
            if self.degree(s) > d:
                if self._warnings:
                    print("However its degree is higher than {}: updating it to {}".format(self.degree(s),d))
                update = True
                self._degrees_dict.pop(s)
            else:
                if self._warnings:
                    print("Its degree is {} which is lower than {}: keeping it that way".format(self._degrees_dict[s],d))
                return

        # check that all faces are in the complex already. If not, warn the user and add faces (recursively)
        faces = s.faces()
        if s.dim>1:
            for f in faces:

                if self._warnings:
                    print("Inserting face {} as well".format(f))
                self.append(f,d)

        if not update:
            self._numSimplices += 1
            self._simplices.append(s)
        else:
            pass

        self._degrees_dict[s] = d
        self._dimension = max(self._dimension,s.dim)
        self._maxDeg = max(self._maxDeg,d)

    def insert(self,l,d):
        self.append(Simplex(l),d)

    def __str__(self):

        return  "\n".join(["{} : {}".format(s,self._degrees_dict[s]) for s in self._simplices])




class ZomorodianCarlsson:
    def __init__(self,filteredComplex,field = 2,strict = True,verbose = False):
        """
        Class for Zomorodian and Carlsson's algorithm for persistent homology.
        Initialization does not compute homology. Call self.computeIntervals
        for the actual computation.
        Arguments:
        - filteredComplex should be an instance of FilteredComplex, on which
        homology will be computed.
        - field: prime number, specifies the field over which homology is
        computed. Default value: 2
        - strict: Boolean. If set to True, homology elements of duration zero,
        i.e. intervals of the form (x,x), will be ignored. Default value: True
        - verbose: Boolean. If set to True, the computeIntervals method will
        output its progress throughout the algo. Default value: False
        """

        self.numSimplices = filteredComplex._numSimplices

        # first, order the simplices in lexico order on dimension, degree and then arbitrary order
        def key(s):
            d = filteredComplex.degree(s)
            return (s.dim,d,s)
        filteredComplex._simplices.sort(key = key)
        self.simplices = filteredComplex._simplices[:]

        # remember the index of each simplex
        self._indexBySimplex = {}
        for i in range(self.numSimplices):
            self._indexBySimplex[self.simplices[i]] = i


        self.dim = filteredComplex._dimension
        self.degrees = filteredComplex._degrees_dict.copy()
        self.field = field


        self.marked = [False for i in range(self.numSimplices)]
        self.T = [None for i in range(self.numSimplices)] # contains couples (index,chain)
        self.intervals = [[] for i in range(self.dim+1)] # contains homology intervals once the algo has finished
        self.pairs = []

        self._maxDeg = filteredComplex._maxDeg
        self._strict = strict
        self._verbose = verbose
        self._homologyComputed = False

    def addInterval(self,k,t,s):
        i = self.degrees[t]
        if not s:
            j = inf
        else:
            j = self.degrees[s]

        if i != j or (not self._strict):
            #if self._verbose:
                #print("Adding {}-interval ({},{})".format(k,i,j))
            self.intervals[k].append((i,j))
            self.pairs.append((t,s))
            #if i == 10:
            #    print(k,t,s)



    def computeIntervals(self):
        """
        Computes the homology group. After running this method,
        homology intervals become accessible with self.getIntervals(k).
        """
        if self._homologyComputed:
            print("Homology was already computed.")
            return
        if self._verbose:
            print("Beginning first pass")
        for j in range(self.numSimplices):
            if j%1000 == 0 and self._verbose:
                print('{}/{}'.format(j,self.numSimplices))
            s = self.simplices[j]
            #if self._verbose:
                #print("Examining {}. Removing pivot rows...".format(s))
            d = self.removePivotRows(s)
            #if self._verbose:
                #print("Done removing pivot rows")
            if d.isEmpty():
                #if self._verbose:
                    #print("Boundary is empty when pivots are removed: marking {}".format(s))
                self.marked[j] = True
            else:

                maxInd = self.maxIndex(d)
                t = self.simplices[maxInd]
                k = t.dim-1
                self.T[maxInd] = (s,d)
                self.addInterval(k,t,s)
                #if self._verbose:
                    #print("Boundary non-reducible: T{} is set to:".format(t))
                    #print(str(d))

        if self._verbose:
            print("First pass over, beginning second pass")
        for j in range(self.numSimplices):
            if j%1000 == 0 and self._verbose:
                print('{}/{}'.format(j,self.numSimplices))
            s = self.simplices[j]
            if self.marked[j] and not self.T[j]:
                k = s.dim -1
                #if self._verbose:
                    #print("Infinite interval found for {}.".format(s))
                self.addInterval(k,s,None)
        if self._verbose:
            print("Second pass over")
        self._homologyComputed = True



    def removePivotRows(self,s):
        d = simplexBoundary(s,self)
        for j in d.coeffs:
            if not self.marked[j]:
                d.coeffs[j] = 0
        d.purge()
        while not d.isEmpty():

            #if self._verbose:
                #print("Current chain d:")
                #print(str(d))

            maxInd = self.maxIndex(d)
            t = self.simplices[maxInd]
            #if self._verbose:
                #print("simplex with max index in d: {} with index {}".format(maxInd))

            if not self.T[maxInd]:
                #if self._verbose:
                    #print("{} is not in T: done removing pivot rows".format(t))
                break

            c = self.T[maxInd][1]
            q = c.getCoeff(t)
            #if self._verbose:
                #print("{} is in T with coeff {}: ".format(t,q),"##########",str(c),"##########",sep='\n'    )
            d = d - pow(q,self.field-2,self.field)*c
        return d


    def maxIndex(self,d):
        currmax = -1
        for j in d.coeffs:
            if j>currmax:
                currmax = j
        return currmax



    def getIntervals(self,d):
        """
        Returns the list of d-dimensional homology elements.
        Can only be run after computeIntervals has been run.
        """
        if not self._homologyComputed:
            print("Warning: homology has not yet been computed. This will return an empty list.")
        return self.intervals[d][:]

    def bettiNumber(self,k,l,p):
        res = 0
        if not self._homologyComputed:
            print("Warning: homology has not yet been computed. This will return 0.")
        for (i,j) in self.intervals[k]:
            if (i <= l and l + p < j ) and p>=0:
                    res+=1
        return res


#%% Plotting functions
import matplotlib.pyplot as plt

def get_min_max(intervals):
	p_min = inf
	p_max = -inf
	for (x,y) in intervals:
		p_min = min(p_min,x)
		p_max = max(p_max,x)
		if y != inf:
			p_max = max(p_max,y)
	return p_min,p_max

def persistence_diagram(intervals,saveAs = None):
	"""
	Plots the persistence diagram of the input list.
	Arguments:
	- intervals: list of tuples (x,y) of reals. y may be inf
	- saveAs: optional. String specifying a path / name to
	save the image of the diagram. If set, the function will not
	show the diagram but only save it.
	"""
	if len(intervals) == 0:
		fig, ax = plt.subplots()
		lower_limit = 0
		upper_limit = 1
		ax.set_xlim(left = lower_limit,right = upper_limit)
		ax.set_ylim(bottom = lower_limit,top = upper_limit)
		ax.plot([lower_limit, upper_limit],[lower_limit , upper_limit],'b-')
		if saveAs:
			plt.savefig(saveAs)
			print("Saved figure at " + saveAs)
		else:
			plt.show()
		return

	p_min, p_max = get_min_max(intervals)
	dp = p_max - p_min

	lower_limit = 0 - dp/5
	upper_limit = p_max + dp/5


	fig, ax = plt.subplots()
	ax.set_xlim(left = lower_limit,right = upper_limit)
	ax.set_ylim(bottom = lower_limit,top = upper_limit)
	ax.plot([lower_limit, upper_limit],[lower_limit , upper_limit],'b-')

	for (x,y) in set(intervals):
		if y == inf:
			ax.plot([x,x],[x,upper_limit],'g-')
		else:
			ax.plot([x],[y],'ro')

	if saveAs:
		plt.savefig(saveAs)
		plt.close()
		print("Saved figure at " + saveAs)
	else:
		plt.show()





def barcode(intervals):
	"""
	Plot the barcode corresponding to the input list.
	"""
	p_min, p_max = get_min_max(intervals)
	dp = p_max - p_min

	lower_limit = p_min - dp/5
	upper_limit = p_max + dp/5

	fig, ax = plt.subplots()
	ax.set_xlim(left = lower_limit,right = upper_limit)

	n = len(intervals)
	for i in range(n):
		(x,y) = intervals[i]
		if x == y:
			ax.plot([x],[y],'ro')
		elif y == inf:
			ax.plot([x,upper_limit],[i,i],'r-')
		else:
			ax.plot([x,y],[i,i],'r-')
	plt.show()
    


#%% Main code

# First step: create a list of simplices and filtration values (also called degrees)
# list_simplex_degree = [([0],0), ([1],0), ([2],1), ([3],1), ([0, 1],1), 
#                        ([1, 2],1), ([0, 3],2), ([2, 3],2), ([0, 2],3), 
#                        ([0, 1, 2],4), ([0, 2, 3],5)]

# # add them one by one in a fresh complex
# fc = FilteredComplex()
# for (simplex, value) in list_simplex_degree:
#     fc.insert(simplex,value)




# # compute persistent homology in Z/2Z
# zc = ZomorodianCarlsson(fc)
# zc.computeIntervals()
# for i in range(2):
#     intervals = zc.getIntervals(i)
#     barcode(intervals)
#     persistence_diagram(intervals)

#%% Plot functions for geometry
import geopandas as gpd

def draw_nodes(ax,graph,nodelist=None,color='red',size=30,alpha=1.0,marker='*'):
    if nodelist == None:
        nodelist = graph.nodes
    d = {'nodes':nodelist,
         'geometry':[Point(graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,marker=marker)
    return ax

def draw_edges(ax,graph,edgelist=None,color='red',width=2.0,style='solid',
               alpha=1.0):
    if edgelist == []:
        return ax
    if edgelist == None:
        edgelist = graph.edges
    d = {'edges':edgelist,
         'geometry':[graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha)
    return ax

def draw_points(ax,points,color='red',size=10,alpha=1.0,marker='o'):
    if len(points) == 0:
        return ax
    if isinstance(points,list):
        d = {'nodes':range(len(points)),
             'geometry':[pt_geom for pt_geom in points]}
    elif isinstance(points,dict):
        d = {'nodes':range(len(points)),
             'geometry':[points[k] for k in points]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha,marker=marker)
    return ax

def draw_lines(ax,lines,color='red',width=2.0,style='solid',alpha=1.0):
    if isinstance(lines,LineString):
        lines = [lines]
    if len(lines) == 0:
        return ax
    d = {'edges':range(len(lines)),
         'geometry':[line_geom for line_geom in lines]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,
                  linestyle=style,alpha=alpha)
    return ax

#%% Graph example
import networkx as nx
from shapely.geometry import Point,LineString

def interpolate_points(lines,num_pts=100,ref='A'):
    points = {}
    for i,line_geom in enumerate(lines):
        dist = 1/num_pts
        start_ind = (i*num_pts)
        for f in range(num_pts+1):
            x,y = line_geom.interpolate(f*dist,normalized=True).xy
            xy = (x[0],y[0])
            points[ref+str(start_ind+f)] = Point(xy)
    return points

# Graph A
edges_A = [(1,2),(2,3)]
cords_A = {1:(-1,0.5), 2:(0,1), 3:(1,0.5)}
edge_geom_A = {e:LineString([cords_A[e[0]],cords_A[e[1]]]) for e in edges_A}

A = nx.Graph()
A.add_edges_from(edges_A)
nx.set_edge_attributes(A,edge_geom_A,'geometry')
nx.set_node_attributes(A,cords_A,'cord')
geom_A = [A.edges[e]['geometry'] for e in A.edges]

# Graph B
edges_B = [(11,12),(12,13),(13,14),(14,15)]
cords_B = {11:(-1,0.25), 12:(-0.5,0.75), 13:(0,1), 14:(0.5,0.6), 15:(1,0.5)}
edge_geom_B = {e:LineString([cords_B[e[0]],cords_B[e[1]]]) for e in edges_B}

B = nx.Graph()
B.add_edges_from(edges_B)
nx.set_edge_attributes(B,edge_geom_B,'geometry')
nx.set_node_attributes(B,cords_B,'cord')
geom_B = [B.edges[e]['geometry'] for e in B.edges]


pts_A = interpolate_points(geom_A,num_pts=20,ref='A')
pts_B = interpolate_points(geom_B,num_pts=20,ref='B')
pts = {}
for pt in pts_A:
    pts[pt] = pts_A[pt]
for pt in pts_B:
    pts[pt] = pts_B[pt]
    

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)

draw_edges(ax1,A,color='red',width=2.0,style='solid')
draw_edges(ax1,B,color='blue',width=2.0,style='solid')
draw_nodes(ax1,A,color='red',size=30,marker='s',alpha=0.6)
draw_nodes(ax1,B,color='blue',size=30,marker='o')

ax2 = fig.add_subplot(1,2,2)
draw_points(ax2,pts_A,color='red',size=15,alpha=0.6,marker='D')
draw_points(ax2,pts_B,color='blue',size=15,alpha=0.6,marker='o')


#%% Main Code with Rips complex
lA = [list(pts_A[pt].coords)[0] for pt in pts_A]
lB = [list(pts_B[pt].coords)[0] for pt in pts_B]

rA = RipsComplex(lA,threshold = 0.23,verbose = True)
rB = RipsComplex(lB,threshold = 0.23,verbose = True)

rA.compute_skeleton(3)
rB.compute_skeleton(3)

zcA = ZomorodianCarlsson(rA.complex, strict = True,verbose = True)
zcB = ZomorodianCarlsson(rB.complex, strict = True,verbose = True)

zcA.computeIntervals()
zcB.computeIntervals()

persistence_diagram(zcA.intervals[0])
persistence_diagram(zcA.intervals[1])
persistence_diagram(zcA.intervals[2])

persistence_diagram(zcB.intervals[0])
persistence_diagram(zcB.intervals[1])
persistence_diagram(zcB.intervals[2])





























