# -*- coding: utf-8 -*-
"""
Pilot for seed generation notebook

Created on Thu Jan 10 16:16:18 2019
"""

import pandas as pd
import numpy as np
from parallel_curves import parallel_curves

def fgen(x, a, b, c):
    #Solve second order polynomials
    form = lambda x: a*x**2 + b*x + c
    return form(x)
    
file = "callosal_curve_coordinates_excel.xlsx"
sheets = ["UpperCurveCoords", "LowerCurveCoords"]
cc_coords = pd.read_excel(file, sheet_name=sheets, header=None)

poly1_cc_2d = cc_coords['UpperCurveCoords'].values[:, 1:]
poly2_cc_2d = cc_coords['LowerCurveCoords'].values[:, 1:]

p = 2 #quadratic
#--- Generate reference curve for anterior corpus callosum ---#
x = np.arange(0, 60, .5)

poly1_x = poly1_cc_2d[:, 0] * -1 #Inverting the "x-axis" so values are positive
poly1_y = poly1_cc_2d[:, 1]

params = np.polyfit(poly1_x, poly1_y, p)
ant_cc = fgen(x, params[0], params[1], params[2])

#--- Generating curves parallel to anterior corpus callosum ---#
res = parallel_curves(x, ant_cc, 5)
params = np.polyfit(res['x_outer'], res['y_outer'], p)
ant_inferior_curve = fgen(x, params[0], params[1], params[2])

res = parallel_curves(x, ant_cc, 15)
params = np.polyfit(res['x_outer'], res['y_outer'], p)
ant_superior_curve = fgen(x, params[0], params[1], params[2])

#--- Generate reference curve for posterior corpus callosum ---#
x = np.arange(6, 40, .5)

poly2_x = poly2_cc_2d[:, 0] * -1
poly2_y = poly2_cc_2d[:, 1]

params = np.polyfit(poly2_x, poly2_y, p)
post_cc = fgen(x, params[0], params[1], params[2])

#--- Generating curves parallel to posterior corpus callosum ---#
res = parallel_curves(x, post_cc, 5)
params = np.polyfit(res['x_outer'], res['y_outer'], p)
post_inferior_curve = fgen(x, params[0], params[1], params[2])

res = parallel_curves(x, post_cc, 15)
params = np.polyfit(res['x_outer'], res['y_outer'], p)
post_superior_curve = fgen(x, params[0], params[1], params[2])