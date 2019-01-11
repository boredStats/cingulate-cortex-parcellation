# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:04:45 2019
"""

import os
import numpy as np
import kmeansUtils as ku
import pandas as pd
import pickle as pkl

import sys
sys.path.append("..")
import proj_utils as pu

with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)

odir = os.path.join(os.getcwd(),"seed_combined")
if not os.path.isdir(odir):
    os.mkdir(odir)

seedList = pData['newNames']

dataFile = r"../neurosynth/output/seed_functional_profiles.csv"
dataDf = pd.read_csv(dataFile, index_col=0)
fpp_df = dataDf.T.corr()

dataFile = r"../jaccard_indices/jaccard_seed_to_networks_raw.csv"
dataDf = pd.read_csv(dataFile, index_col=0,)
network_df = dataDf.T.corr()

combined_matrix = np.ndarray(shape=[fpp_df.values.shape[0], 2*fpp_df.values.shape[0]])
combined_matrix[:, :58] = network_df.values
combined_matrix[:, 58:] = fpp_df.values
combined_colnames = list(fpp_df.columns.values) + list(network_df.columns.values)
combined_df = pd.DataFrame(combined_matrix, index=fpp_df.index, columns=combined_colnames)

n_clusters = combined_df.values.shape[0]-1
km_data, sil_data = ku.kmeans(combined_df.values, n_clusters)

dump_data= {}
dump_data['km'] = km_data
dump_data['sil'] = sil_data
with open(os.path.join(odir, "results.pkl"), 'wb') as file:
    pkl.dump(dump_data, file)
    
fname = os.path.join(odir, "silScores.png")
best_k = ku.bestSilScores(sil_data, 28, fname)

#bestK = 4 #overwriting best_k for testing
network_consistency = ku.bestSolution(km_data, best_k, seedList)

fname = os.path.join(odir, "%dkSolution.csv" % best_k)
network_consistency.to_csv(fname, header=False)

palette = pData['catColors']
coords = pData['coordsMNI']
colors = pu.node_color_generator(palette, network_consistency.values[:, 0])
fname = os.path.join(odir, "%dkSolutionBrain.png" % best_k)
pu.plotBrains(coords, colors, fname)