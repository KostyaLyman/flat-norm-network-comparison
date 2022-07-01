# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:36:52 2022

@author: rm5nz
"""

from __future__ import absolute_import

import numpy as np
from scipy import sparse


from cvxopt import matrix, spmatrix, solvers

solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-9
solvers.options['feastol'] = 1e-10
solvers.options['show_progress'] = False

def lp_solver(c, cons, b):
    '''
    Linear program solver. 
    :param float c: a vector of cost coefficients for the objective function.
    :param int cons: a constraint matrix A of dimension kmx(2m+k(2m+2n)).
    :param int b: a vector of coefficients.
    :returns: sol_x, objective_value -- the optimal solution and objective value resp.
    :rtype: int, float
    '''
    g = -sparse.identity(len(c), dtype=np.int8, format='coo')
    h = np.zeros(len(c))
    G = spmatrix(g.data.tolist(), g.row, g.col, g.shape,  tc='d')
    h = matrix(h)
    c = matrix(c)
    cons = spmatrix(cons.data.tolist(), cons.row, cons.col, cons.shape, tc='d')
    b = matrix(b,(len(b),1),'d')
    sol = solvers.lp(c, G, h, cons, b, solver='glpk')
    sol_x = np.array(sol['x'])
    objective_value = sol['primal objective']
    return sol_x, objective_value