# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 09:11:37 2018

Multiple factor analysis
"""
import sys
sys.path.append("..")
import proj_utils as pu

import os
import prince
import numpy as np
import pandas as pd
import pickle as pkl
import efaUtils as utils
from os.path import join, isdir

with open(r"../pData.pkl", "rb") as f:
    pData = pkl.load(f)
seedList = pData['newNames']
catColors = pData['catColors']
coords = pData['coordsMNI']
coordsDf = pd.DataFrame(coords, index=seedList)

fcluster = r"../k_means/seed_fpp/3kSolution.csv"
fclusterDf = pd.read_csv(fcluster, header=None, index_col=0)

jcluster = r"../k_means/seed_networks/3kSolution.csv"
jclusterDf = pd.read_csv(jcluster, header=None, index_col=0)

odir = join(os.getcwd(), "mfa")
if not isdir(odir):
    os.mkdir(odir)

jaccardFile = r"../jaccard_indices/jaccard_seed_to_networks_raw.csv"
jaccard_df = pd.read_csv(jaccardFile, index_col=0)
corrDf = jaccard_df.T.corr()
dataJ = utils.centerMat(corrDf.values)
colJ = ["icaNet_%s" % s for s in seedList]

file = r"../neurosynth/output/seed_functional_profiles.csv"
fileDf = pd.read_csv(file,index_col=0)
corr_prep = pd.DataFrame(fileDf.values.T)
corrDf = corr_prep.corr()
dataF = utils.centerMat(corrDf.values)
colF = ["fpp_%s" % s for s in seedList]

X = np.ndarray(shape=[len(dataJ), 2*len(dataJ)])
X[:, :58] = dataJ
X[:, 58:] = dataF
colnames = colJ + colF

Z = pd.DataFrame(X, index=seedList, columns=colnames)
eigs, p = utils.permPCA(Z.values)
utils.plotScree(eigs, p, kaiser=False, fname=join(odir, "scree.png"))

groups = {}
groups['icaNetworks'] = [c for c in list(Z) if "icaNet" in c]
groups['fpp'] = [c for c in list(Z) if "fpp" in c]

mfa = prince.MFA(groups=groups,
                 n_components=4,
                 n_iter=3,
                 copy=True,
                 check_input=False,
                 engine='auto')
mfa.fit(Z)
eigs = mfa.eigenvalues_
loadings = mfa.column_correlations(Z)
sqloadings = loadings**2
loadings.to_csv(join(odir, "loadings.csv"))
sqloadings.to_csv(join(odir, "sqloadings.csv"))
factorScores = mfa.row_coordinates(Z)
factorScores.to_csv(join(odir, "factorScores.csv"))

f = r"../k_means/combinedNetworkFPP/3kSolution.csv"
kCombined = pd.read_csv(f, index_col=0, header=None)
col = pu.node_color_generator(catColors, kCombined.values)

fs = factorScores.values[:, 0:2]
#utils.plotCircleOfCorrMFA(fs, eigs, col=col, fname=join(odir, "fs.png"))
fsn = utils.normTo1(fs)
utils.plotFS(fsn, eigs, 'c', col=col, fname=join(odir, "fs_normalized.png"))
R, theta, rgb = utils.createColorSpace(fsn)
rgb[rgb > 1] = 1
utils.createColorLegend(R, theta, join(odir, "fs_colorLegend.png"))
utils.plotBrains(pData['coordsMNI'], rgb, join(odir, "fs_brain.png"))

#--- ica networks ---#
icaNetLoadings = loadings[loadings.index.str.contains("icaNet")]

reorderedSol = []
for sf in list(icaNetLoadings.index):
    for i,sl in enumerate(list(jclusterDf.index)):
        if sl in sf:
            reorderedSol.append(jclusterDf.values[i, 0])
hcols1 = pu.node_color_generator(catColors, np.asarray(reorderedSol))

floads = icaNetLoadings.values[:,0:2]
R, theta, rgb = utils.createColorSpace(floads)
utils.createColorLegend(R, theta, join(odir, "icaNet_colorLegend.png"))
utils.plotBrains(pData['coordsMNI'], rgb, join(odir, "icaNet_brain.png"))

fname = join(odir, "icaNet_circleOfCorr12.png")
utils.plotFS(floads, eigs, col=hcols1, fname=fname)

#--- FPP ---#
fppLoadings = loadings[loadings.index.str.contains("fpp")]

reorderedSol = []
for sf in list(icaNetLoadings.index):
    for i,sl in enumerate(list(fclusterDf.index)):
        if sl in sf:
            reorderedSol.append(fclusterDf.values[i])
hcols2 = pu.node_color_generator(catColors, np.asarray(reorderedSol))

floads = fppLoadings.values[:,0:2]
R, theta, rgb = utils.createColorSpace(floads)
utils.createColorLegend(R, theta, join(odir, "fpp_colorLegend.png"))
utils.plotBrains(pData['coordsMNI'], rgb, join(odir, "fpp_brain.png"))

fname = join(odir, "fpp_circleOfCorr12.png")
utils.plotFS(floads, eigs, col=hcols2, fname=fname)

t = np.zeros(loadings.values.shape[0])
t[58:] = 1
color = [hcols1] + [hcols2]

#fname = join(odir, "test.png")
#utils.plotFS(loadings.values[:, 0:2], eigs, t, col=color, fname=fname)