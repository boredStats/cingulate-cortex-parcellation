# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:10:43 2018

Creating functional profiles using all seeds

@author: ixa080020
"""

import os
import pandas as pd
import numpy as np
import pickle as pkl
import neurosynth_utils as nu

from sklearn.naive_bayes import GaussianNB
from classification import RegionalClassifier
from sklearn.metrics import roc_auc_score

odir = os.path.join(os.getcwd(),"output")
if not os.path.isdir(odir):
    os.mkdir(odir)

with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)
seed_list = pData['newNames']

print("Loading neurosynth data...")
dataset, nicknames, word_keys = nu.functional_preference_profile_prep()

mask_nifti = r"../create_seeds/seed_files/all_seeds_cluster_indexed_nifti.nii"
print("Running classifier...") #nibabel 1.3.0 req
clf = RegionalClassifier(dataset, mask_nifti, GaussianNB()) 
clf.classify(scoring=roc_auc_score)

fn = pd.merge(pd.DataFrame(clf.feature_names, columns=['topic_name']),
              word_keys)['Topic name'].tolist()
feat_df = clf.get_formatted_importances(feature_names=fn)

# Getting full functional profiles from each seed
all_feats = list(feat_df['feature'].values)

unique_feats = nu.get_unique_features(all_feats)
unique_regions = np.unique(feat_df['region'].values)

#creating dataframe of functional profiles for each seed
profile_values = nu.reorganize_feature_df(feat_df,unique_regions,unique_feats)
profile_df = pd.DataFrame(profile_values,index=seed_list,columns=unique_feats)

profile_df.drop(labels=["blank"],axis=1,inplace=True)
profile_df.to_csv(os.path.join(odir,"seed_functional_profiles.csv"))