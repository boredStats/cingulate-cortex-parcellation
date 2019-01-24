# -*- coding: utf-8 -*-
"""
Calculate Jaccard Similarity Index for S2V maps and ICA-networks
Local version

Created on Wed Jan  9 13:37:45 2019
"""

import os
import nilearn
import pandas as pd
import pickle as pkl
import numpy as np
import nibabel as nib
from nilearn import image, masking
from sklearn.metrics import jaccard_similarity_score as jsc
nilearn.EXPAND_PATH_WILDCARDS = False

with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)
    
network_names = pData['network_names']
seed_list = pData['oldNames']

parent_dir = pData['server_path']
res_dir = os.path.join(parent_dir,"projects/conn_Cingulate_project/results")
seed_dir = os.path.join(res_dir, "secondlevel/ANALYSIS_01/AllSubjects/rest")
network_dir = r"./network_masks"

#Create list of s2v folder results 
s2v_folder_list = [os.path.join(seed_dir, s) for s in seed_list]

#Create list of thresholded, unthresholded s2v masks
thresh_masks = []
unthresh_masks = []
for s2v_folder in s2v_folder_list:
    thresh_fname = "thresh_con_fdr05.Mask.img" 
    img_file = [f for f in os.listdir(s2v_folder) if thresh_fname in str(f)]
    thresh_masks.append(os.path.join(s2v_folder,img_file[0]))
	
    unthresh_fname = "mask.nii"
    img_file = [f for f in os.listdir(s2v_folder) if unthresh_fname in str(f)]
    unthresh_masks.append(os.path.join(s2v_folder,img_file[0]))

#Create list of ICA networks masks
network_masks = []
for x in network_names:
    network_fname = [f for f in os.listdir(network_dir) if x in f][0]
    img_file = os.path.join(network_dir, network_fname)
    network_masks.append(img_file)

#Resampling network masks to match s2v thresh mask dimensions
target_img = thresh_masks[0]
resamp_network_masks = image.resample_to_img(network_masks,target_img)

#Making sure that brain comparisons are being made across the same mask
#Creating stack of unthresholded s2v masks
unthresh_mask_data = np.zeros(shape=[91,109,91,58])
k = 0
for x in unthresh_masks:
    unthresh_file = nib.load(x)
    unthresh_mask_data[:,:,:,k] = unthresh_file.get_data()
    k+=1
#Creating composite whole brain mask from s2v unthresholded masks 
composite_mask = np.zeros(shape=[91,109,91])
for x in range(91):
    for y in range(109):
        for z in range(91):
            if any(unthresh_mask_data[x,y,z,:]):
                composite_mask[x,y,z] = 1
                
#Creating whole brain nifti object, resampling to thresholded mask dims
unthresh_img = nib.load(unthresh_masks[0])
thresh_img = nib.load(thresh_masks[0])
unthresh_final_mask_nifti = nib.Nifti1Image(
        composite_mask,
        unthresh_img.affine,
        unthresh_img.header)
resamp_composite = image.resample_img(
        unthresh_final_mask_nifti,
        target_affine=thresh_img.affine,
        target_shape=thresh_img.shape)
#Making sure that values are only zero or one
for x in range(thresh_img.shape[0]):
    for y in range(thresh_img.shape[1]):
        for z in range(thresh_img.shape[2]):
            if resamp_composite.get_data()[x,y,z] > 0:
                resamp_composite.get_data()[x,y,z] = 1

#Vectoring volume data            
network_vectors = masking.apply_mask(resamp_network_masks, resamp_composite)
s2v_vectors = masking.apply_mask(thresh_masks, resamp_composite)

#Calculating jaccard similarity profiles of each seed by comparing their
#spatial similarity to conn toolbox's resting-state ICA networks
jaccard_results = np.zeros(shape=[58,8])
for x in range(58):
    s2v_vect = s2v_vectors[x,:]
    for y in range(8):
        network_vect = network_vectors[y,:]
        network_vect[network_vect<1] = -1
        
        jaccard_index = jsc(s2v_vect,network_vect)
        jaccard_results[x,y] = jaccard_index

res_df = pd.DataFrame(jaccard_results, index=seed_list, columns=network_names)
res_df.to_csv("jaccard_seed_to_networks_raw.csv")
np.savetxt("jaccard_to_networks_raw.txt",jaccard_results,fmt="%.18f")

##Calculating correlation between seed profiles
#jaccard_df = pd.DataFrame(np.transpose(jaccard_results))
#corr_matrix = jaccard_df.corr()
#np.savetxt('jaccard_to_networks_corr.txt',corr_matrix.values,fmt='%.18f')