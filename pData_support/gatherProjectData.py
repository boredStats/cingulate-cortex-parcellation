# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:36:11 2018

gather project data

@author: ixa080020
"""

import numpy as np
import pandas as pd
import pickle as pkl

import sys
sys.path.append('..')
import proj_utils as pu

projData = pu._proj_data().get_data()

with open(r"./serverPath.txt", 'r') as f:
    server_path = f.read()

seedNameDf = pd.read_table(r"./renamed_seeds_Vogt2005_mod.txt")
seedList = list(seedNameDf['New Names'].values)

catColors = np.loadtxt(r"./rgb_google20c.txt", delimiter=',')
catColors = np.divide(catColors, 255)

with open(r"./network_names.txt" ,'r') as f:
    network_names = [s for s in f.read().split('\n')]

pData = {}
pData['coordsMNI'] = projData['coords']
pData['oldNames'] = projData['seed_list']
pData['newNames'] = seedList
pData['icaNetworks'] = projData['network_names']
pData['tabColors'] = projData['palette']
pData['catColors'] = catColors
pData['server_path'] = server_path
pData['network_names'] = network_names

with open(r"./../pData.pkl","wb") as f:
    pkl.dump(pData, f)