# -*- coding: utf-8 -*-
"""
Utility functions for neurosynth functional preference profile-related stuff

Created on Wed Jan 16 13:13:29 2019
"""
import numpy as np
import pandas as pd

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