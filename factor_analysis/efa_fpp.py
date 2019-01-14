# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:33:46 2018

Code for exploratory factor analysis - functional preference profiles
"""
import os
import numpy as np
import pandas as pd
import pickle as pkl
import efaUtils as utils
from factor_analyzer import FactorAnalyzer
from os.path import join, isdir
from os import getcwd, mkdir

import sys
sys.path.append("..")
import proj_utils as pu

with open(r"../pData.pkl", "rb") as f:
    pData = pkl.load(f)
seedList = pData['newNames']
catColors = pData['catColors']
coords = pData['coordsMNI']

cdir = getcwd()
odir = join(cdir, "fpp")
if not isdir(odir):
    mkdir(odir)
    
cpairs = []
pairings = np.arange(0, catColors.shape[0], 2)
for i in pairings:
    cpairs.append([catColors[i, :] , catColors[i+1, :]])

clusterSol = r"../k_means/seed_fpp/3kSolution.csv"
clustSolDf = pd.read_csv(clusterSol, header=None, index_col=0)
hcols = pu.node_color_generator(catColors, (clustSolDf.values[:, 0]))

#--- using FPP correlation matrix matrix ---#
file = r"../neurosynth/output/seed_functional_profiles.csv"
fileDf = pd.read_csv(file,index_col=0)

corr_prep = pd.DataFrame(fileDf.values.T)
corrDf = corr_prep.corr()
data = utils.centerMat(corrDf.values)
dataDf = pd.DataFrame(data, columns=seedList, index=seedList)

#--- Running PCA to determine number of factors ---#
eigs, p, percent = utils.permPCA(dataDf.values)
utils.plotScree(eigs, p, kaiser=False, fname=join(odir, "scree.svg"))

#--- Running FA ---#
rot = None
fa = FactorAnalyzer()
fa.analyze(dataDf, 3, rot)
loadings = fa.loadings
loadings.to_csv(join(odir, "%s_loadings.csv" % str(rot)))
sqloadings = loadings**2
sqloadings.to_csv(join(odir, "%s_sqloadings.csv" % str(rot)))

f = loadings.values[:, 0:2]
orig_eigs, com_eigs = fa.get_eigenvalues()
e = np.ndarray.flatten(com_eigs.values)
percent = (np.multiply(100, e)) / np.sum(e)

eig_cols = ['eigenvalues', 'p_value', 'percent_variance_explained']
eig_df = pd.DataFrame(columns=eig_cols)
eig_df[eig_cols[0]] = e
eig_df[eig_cols[1]] = p
eig_df[eig_cols[2]] = percent
eig_df.to_csv(os.path.join(odir, 'eigenvalues.csv'))

R, theta, rgb = utils.createColorSpace(f)
utils.createColorLegend(R, theta, join(odir, "%s_colorLegend.svg" % str(rot)))
utils.plotBrains(coords, rgb, join(odir, "%s_brain.svg" % str(rot)))
fn = join(odir, "%s_circleCorr.svg") % str(rot)
utils.plotFS(f, e, 'l', col=hcols, fname=fn)

rot = 'varimax'
fa = FactorAnalyzer()
fa.analyze(dataDf, 3, rot)
loadings = fa.loadings
loadings.to_csv(join(odir, "%s_loadings.csv" % str(rot)))
sqloadings = loadings**2
sqloadings.to_csv(join(odir, "%s_sqloadings.csv" % str(rot)))

f = loadings.values[:, 0:2]
orig_eigs, com_eigs = fa.get_eigenvalues()
e = np.ndarray.flatten(com_eigs.values)

R, theta, rgb = utils.createColorSpace(f)
utils.createColorLegend(R, theta, join(odir, "%s_colorLegend.svg" % str(rot)))
utils.plotBrains(coords, rgb, join(odir, "%s_brain.svg" % str(rot)))
fn = join(odir, "%s_circleCorr.svg") % str(rot)
utils.plotFS(f, e, 'l', col=hcols, fname=fn)