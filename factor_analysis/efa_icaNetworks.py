# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:33:46 2018

Code for exploratory factor analysis - jaccard ica networks
"""
import os
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
import efaUtils as utils
from factor_analyzer import FactorAnalyzer
from os.path import join

with open(r"../pData.pkl", "rb") as f:
    pData = pkl.load(f)
seedList = pData['newNames']
catColors = pData['catColors']

clusterSol = r"../k_means/seed_networks/3kSolution.csv"
clustSolDf = pd.read_csv(clusterSol, header=None, index_col=0)
hcols = pu.node_color_generator(catColors, (clustSolDf.values[:, 0]))

cpairs = []
pairings = np.arange(0, catColors.shape[0], 2)
for i in pairings:
    cpairs.append([catColors[i, :] , catColors[i+1, :]])
    
odir = r"./networks"
if not os.path.isdir(odir):
    os.mkdir(odir)

#--- using Jaccard correlation matrix ---#
jaccardFile = r"../jaccard_indices/jaccard_seed_to_networks_raw.csv"
jaccardRaw = pd.read_csv(jaccardFile, index_col=0)
jaccardCorr = jaccardRaw.T.corr()
data = utils.centerMat(jaccardCorr.values)
dataDf = pd.DataFrame(data, columns=seedList, index=seedList)

#--- Running PCA to determine number of factors ---#
eigs, p = utils.permPCA(dataDf.values)
utils.plotScree(eigs, p, kaiser=False, fname=join(odir, "scree.png"))

#--- Running eFA ---#
rotation = None
rotName = 'noRotation'

fa = FactorAnalyzer()
fa.analyze(dataDf, 3, rotation)
loadings = fa.loadings
sqloadings = loadings**2
loadings.to_csv(join(odir, "%s_loadings.csv" % rotName))
sqloadings.to_csv(join(odir, "%s_sqloadings.csv" % rotName))

R, theta, rgb = utils.createColorSpace(loadings.values)
utils.createColorLegend(R, theta, join(odir, "%s_colorLegend.png" % rotName))
utils.plotBrains(pData['coordsMNI'], rgb, join(odir, "%s_brain.png" % rotName))

fname = join(odir, "%s_circleOfCorr12.png") % rotName
ls = loadings.values[:, 0:2]
utils.plotFS(ls, eigs, 'c', col=hcols, fname=fname)

#fname = join(odir, "%s_circleOfCorr23.png") % rotName
#utils.plotCircleOfCorr(fa, [2, 3], hcols, fname=fname)
#fname = join(odir, "%s_circleOfCorr13.png") % rotName
#utils.plotCircleOfCorr(fa, [1, 3], hcols, fname=fname)

#for i in range(len(list(loadings))):
#    rgba = utils.generateLoadColors(cpairs[i], loadings.values[:, i])
#    fname = join(odir, "%s_brainFactor%d.png" % (rotName, (i+1)))
#    utils.plotBrains(pData['coordsMNI'], rgba, fname)

#--- Rotating subspace ---#
rotation = 'varimax'
rotName = rotation

fa.analyze(dataDf, 3, rotation)
loadings = fa.loadings
sqloadings = loadings**2
loadings.to_csv(join(odir, "%s_loadings.csv" % rotName))
sqloadings.to_csv(join(odir, "%s_sqloadings.csv" % rotName))

R, theta, rgb = utils.createColorSpace(loadings.values)
utils.createColorLegend(R, theta, join(odir, "%s_colorLegend.png" % rotName))
utils.plotBrains(pData['coordsMNI'], rgb, join(odir, "%s_brain.png" % rotName))

fname = join(odir, "%s_circleOfCorr12.png") % rotName
ls = loadings.values[:, 0:2]
utils.plotFS(ls, eigs, 'c', col=hcols, fname=fname)

#fname = join(odir, "%s_circleOfCorr23.png") % rotName
#utils.plotCircleOfCorr(fa, [2, 3], hcols, fname=fname)
#fname = join(odir, "%s_circleOfCorr13.png") % rotName
#utils.plotCircleOfCorr(fa, [1, 3], hcols, fname=fname)

#for i in range(len(list(loadings))):
#    rgba = utils.generateLoadColors(cpairs[i], loadings.values[:, i])
#    fname = join(odir, "%s_brainFactor%d.png" % (rotName, (i+1)))
#    utils.plotBrains(pData['coordsMNI'], rgba, fname)