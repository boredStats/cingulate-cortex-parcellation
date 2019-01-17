# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:36:11 2018

gather project data

@author: ixa080020
"""
import os
import numpy as np
import pandas as pd
import pickle as pkl

import sys
sys.path.append('..')
import proj_utils as pu

pdir = pu._get_proj_dir_v2()
projData = pu.proj_data().get_data()

seedNameDf = pd.read_table(os.path.join(pdir,"renamed_seeds_Vogt2005_mod.txt"))
seedList = list(seedNameDf['New Names'].values)

catColors = np.loadtxt("./rgb_google20c.txt", delimiter=',')
catColors = np.divide(catColors, 255)

pData = {}
pData['coordsMNI'] = projData['coords']
pData['oldNames'] = projData['seed_list']
pData['newNames'] = seedList
pData['icaNetworks'] = projData['network_names']
pData['tabColors'] = projData['palette']
pData['catColors'] = catColors

with open(r"./../pData.pkl","wb") as f:
    pkl.dump(pData, f)