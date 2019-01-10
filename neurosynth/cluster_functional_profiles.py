# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:31:45 2018

creating functional profiles using major clusters

NOTE: Nibabel 1.3.0 is required to run this script
Jan 10 edit - now it requres 2.0.2???

@author: ixa080020
"""

import os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.naive_bayes import GaussianNB
from classification import RegionalClassifier
from sklearn.metrics import roc_auc_score

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

with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)
seed_list = pData['newNames']

odir = os.path.join(os.getcwd(),"output")
if not os.path.isdir(odir):
    os.mkdir(odir)

hier_cluster_file = r"../create_seeds/hierarchical_clustering/hierarchical_cluster_nifti.nii"
#kmeans_cluster_file = os.path.join(pdir,"create_seeds/)

print("Calculating functional preference profiles...")
dataset, nicknames, word_keys = functional_preference_profile_prep()
clf = RegionalClassifier(dataset,hier_cluster_file, GaussianNB()) #nibabel 1.3.0 req
clf.classify(scoring=roc_auc_score)

feature_names = pd.merge(
        pd.DataFrame(clf.feature_names,columns=['topic_name']),
        word_keys)['Topic name'].tolist()

fn = pd.merge(pd.DataFrame(clf.feature_names, columns=['topic_name']), word_keys)['Topic name'].tolist()
feat_df = clf.get_formatted_importances(feature_names=fn)
"""
Getting full functional profiles from each cluster
removed save to csv section, see next chunk 
"""
#print("Saving functional preference profile data...")
cluster_names = ['hclust1','hclust2','hclust3']

all_feats = list(feat_df['feature'].values)
unique_feats = get_unique_features(all_feats)
unique_regions = np.unique(feat_df['region'].values)

#creating dataframe of functional profiles for each cluster
profile_values = reorganize_feature_df(feat_df,unique_regions,unique_feats)
profile_df = pd.DataFrame(profile_values,columns=unique_feats,index=cluster_names)

profile_df.drop(labels=["blank"],axis=1,inplace=True)
#profile_df.to_csv(os.path.join(odir,"cluster_functional_profiles.csv"))

"""
Getting log-odds ratio of each functional profile, bootstrapped results
Nov 28 2018
"""
from classification import permute_log_odds, bootstrap_log_odds
#lor_z = permute_log_odds(clf, 100, feature_names=nicknames.nickname, region_names=cluster_names)
lor_z = permute_log_odds(clf, 100, feature_names=word_keys['Topic name'], region_names=cluster_names)

topic_list = [t for t in unique_feats if "blank" not in t]
#ut = proj_utils.get_unique_features(topic_list)
select_ps = lor_z[lor_z['Topic name'].isin(topic_list)] #TO-DO:get topic list 

from statsmodels.sandbox.stats.multicomp import multipletests
reject, p_corr, a, a1 = multipletests(select_ps.p, alpha=0.01, method='fdr_tsbky')

select_ps['reject_01'] = reject & (select_ps.lor_z > 0) # Was the null hypothesis rejected?
select_ps['p_corr_01'] = p_corr # Adjusted p-value

lor_ci = bootstrap_log_odds(clf, 100, feature_names=word_keys['Topic name'], region_names=cluster_names)
select_ci = lor_ci[lor_ci.topic_name.isin(topic_list)]

fname = os.path.join(odir, "cluster_functional_profiles.xlsx")
### Saving all results to Excel
writer = pd.ExcelWriter(fname, engine="xlsxwriter")
profile_df.to_excel(writer, sheet_name='cluster FPP')
select_ps.to_excel(writer, sheet_name='significance testing')
select_ci.to_excel(writer, sheet_name='confidence intervals')
writer.save()