# -*- coding: utf-8 -*-
"""
k-means on functional preference profiles

Created on Thu Jan 10 20:01:32 2019
"""

import os
import kmeansUtils as ku
import pandas as pd
import pickle as pkl

import sys
sys.path.append("..")
import proj_utils as pu

with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)
seedList = pData['newNames']

odir = os.path.join(os.getcwd(),"seed_fpp")
if not os.path.isdir(odir):
    os.mkdir(odir)

#dataFile = r"../neurosynth/output/seed_functional_profiles.csv"
#dataDf = pd.read_csv(dataFile, index_col=0)
#corrDf = dataDf.T.corr()
#
#n_clusters = len(corrDf.values)-1
#km_data, sil_data = ku.kmeans(corrDf.values, n_clusters)
#
#dump_data= {}
#dump_data['km'] = km_data
#dump_data['sil'] = sil_data
#with open(os.path.join(odir, "results.pkl"), 'wb') as file:
#    pkl.dump(dump_data, file)

with open(os.path.join(odir, "results.pkl"), 'rb') as file:
    dump_data = pkl.load(file)
sil_data = dump_data['sil']
km_data = dump_data['km']

fname = os.path.join(odir, "silScores.svg")
best_k = ku.bestSilScores(sil_data, 28, fname)

#bestK = 4 #overwriting best_k for testing
network_consistency = ku.bestSolution(km_data, best_k, seedList)

fname = os.path.join(odir, "%dkSolution.csv" % best_k)
network_consistency.to_csv(fname, header=False)

palette = pData['catColors']
coords = pData['coordsMNI']
colors = pu.node_color_generator(palette, network_consistency.values[:, 0])
fname = os.path.join(odir, "%dkSolutionBrain.svg" % best_k)
pu.plotBrains(coords, colors, fname)