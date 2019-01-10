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

"""
These functions have been copied to proj_utils, keeping here for posterity
"""
def get_unique_features(feature_list):
    """
    Function for getting unique features from feature dataframe output
    """
    unique_feats = []
    for i,f in enumerate(feature_list):
        if str(f) == 'nan':
            feature_list[i] = 'blank'
    for index,feat in enumerate(feature_list):
        if index != len(feature_list)-1:
            nextFeat = str(feature_list[index+1])
        if feat != nextFeat:
            unique_feats.append(feat)
    return unique_feats

def reorganize_feature_df(feat_df,unique_regions,unique_feats):
    """
    Function for creating a better feature matrix from de la Vega's feature dataframe
    """
    profile_values = np.ndarray(shape=[len(unique_regions),len(unique_feats)])
    for indf,row in feat_df.iterrows():
        row_region = row['region']
        row_feat = row['feature']
        for inr,region in enumerate(unique_regions):
            for inf,feature in enumerate(unique_feats):
                if str(row_region)==str(region) and str(row_feat)==str(feature):
                    profile_values[inr,inf] = row['importance']  
    return profile_values

def functional_preference_profile_prep():
	"""
	Function for extracting functional preference profile data
	"""
	from neurosynth.base.dataset import Dataset
	dataset = Dataset.load("data/neurosynth_60_0.4.pkl")

	nicknames = pd.read_csv('data/v4-topics-60.txt', delimiter='\t')
	nicknames['topic_name'] = nicknames.apply(lambda row: '_'.join([str(row.topic_number)] + row.top_words.split(' ')[0:3]), axis=1)
	nicknames = nicknames.sort_values('topic_name')

	word_keys = pd.read_csv("data/topic_keys60-july_cognitive.csv")
	word_keys['top_2'] = word_keys['Top words'].apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
	word_keys['topic_name'] = "topic" + word_keys['topic'].astype('str')
	
	return dataset,nicknames,word_keys

"""
de la Vega stuff (modified)
"""
#from neurosynth.base.dataset import Dataset
from sklearn.naive_bayes import GaussianNB
from classification import RegionalClassifier
from sklearn.metrics import roc_auc_score

print("Loading neurosynth data...")
dataset, nicknames, word_keys = functional_preference_profile_prep()
"""
Functional profiles using seeds individually
"""
odir = os.path.join(os.getcwd(),"output")
if not os.path.isdir(odir):
    os.mkdir(odir)

with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)
seed_list = pData['newNames']

mask_nifti = r"../create_seeds/seed_files/all_seeds_cluster_indexed_nifti.nii"
print("Running classifier...") #nibabel 1.3.0 req
clf = RegionalClassifier(dataset,mask_nifti, GaussianNB()) 
clf.classify(scoring=roc_auc_score)

fn = pd.merge(pd.DataFrame(clf.feature_names, columns=['topic_name']), word_keys)['Topic name'].tolist()
feat_df = clf.get_formatted_importances(feature_names=fn)

# Getting full functional profiles from each seed
all_feats = list(feat_df['feature'].values)

unique_feats = get_unique_features(all_feats)
unique_regions = np.unique(feat_df['region'].values)

#creating dataframe of functional profiles for each seed
profile_values = reorganize_feature_df(feat_df,unique_regions,unique_feats)
profile_df = pd.DataFrame(profile_values,index=seed_list,columns=unique_feats)

profile_df.drop(labels=["blank"],axis=1,inplace=True)
profile_df.to_csv(os.path.join(odir,"seed_functional_profiles.csv"))