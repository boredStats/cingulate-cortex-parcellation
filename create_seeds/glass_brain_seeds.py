# -*- coding: utf-8 -*-
"""
Create glass brain figures for seeds

Created on Thu Jan 10 10:16:02 2019
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from nilearn.plotting import plot_connectome

def plotBrains2(coords, colors, fname=None):
    #Simplified from projUtils version
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    numc = len(colors)
    adjMat = np.zeros(shape=(numc, numc))
    fig = plt.figure(figsize=[24, 12])
    plot_connectome(adjMat, coords, colors, display_mode='lr', node_size=900,
                    figure=fig, output_file=fname, black_bg=False)
    plt.show()

def vogt_colormaker(name, catColors):
    keys = ["sgACC", "pgACC", "aMCC", "pMCC", "dPCC", "vPCC", "RSC"]
    return [catColors[i] for i,k in enumerate(keys) if name == k][0]

with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)
    
old_list = pData['oldNames']
coords = pData['coordsMNI']
catColors = pData['catColors']

c = .3
grey30 = np.asarray([c, c, c])
c = .6
grey80 = np.asarray([c, c, c])

node_colors = []
for i in old_list:
    if "ACC" in i:
        node_colors.append(grey80)
    else:
        node_colors.append(grey30)

plotBrains2(coords, node_colors, r"./seed_brain_grey.png")
plotBrains2(coords, node_colors, r"./seed_brain_grey.svg")

new_list = pData['newNames']
names = [i.split("_")[0] for i in new_list]
vogt_colors = np.asarray([vogt_colormaker(n, catColors) for n in names])

plotBrains2(coords, vogt_colors, r"./seed_brain_vogt.png")
plotBrains2(coords, vogt_colors, r"./seed_brain_vogt.svg")