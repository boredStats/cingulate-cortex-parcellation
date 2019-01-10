# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:38:23 2018

create nifti files based on k-means solutions
"""

import os
import pickle as pkl
import nibabel as nib
import numpy as np
import pandas as pd

output_dir = os.path.join(os.getcwd(),"k_means")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    
with open(r"../pData.pkl", 'rb') as f:
    pData = pkl.load(f)
old_names = pData['oldNames']

kclust_dir = r"../k_means"
sol_fpath = os.path.join(kclust_dir,"seed_networks/3kSolution.csv")
sol_t = pd.read_csv(sol_fpath,index_col=0,header=None)
sol_cleaned = sol_t.values[:, 0]
sol_df = pd.DataFrame(sol_cleaned,index=old_names)

seed_dir = os.path.join(os.getcwd(),"seed_files/nifti")
imgs = os.listdir(seed_dir)
"""
single nifti
""" 
target_vol = nib.load(os.path.join(seed_dir,imgs[0]))
target_data = target_vol.get_data()
target_vect = np.ndarray.flatten(target_data)

final_vect = np.zeros(shape=target_vect.shape)

for cluster in range(3):
    for index,seed in enumerate(list(sol_df.index)):
        for img in imgs:
            vol = nib.load(os.path.join(seed_dir,img))
            img_data = vol.get_data()
            data_vect = np.ndarray.flatten(img_data)
            
            if int(sol_df.values[index])==cluster+1 and seed in img:
                true_label = cluster+1
                final_vect[data_vect==1] = true_label

final_brain = np.reshape(final_vect,newshape=target_data.shape)
mask_nifti = nib.Nifti1Image(final_brain,affine=target_vol.affine)
nib.save(mask_nifti,os.path.join(output_dir,"kmeans_indexed.nii.gz"))
"""
separate nifti files
"""
unique_sol = np.unique(sol_df.values)

seed_dir = os.path.join(os.getcwd(),"seed_files/nifti")
clustered_seed_files = []
for i in range(len(unique_sol)):
    temp = []
    for ir,rows in enumerate(list(sol_df.index)):
        for sf in os.listdir(seed_dir):
            if rows in sf and i==int(sol_df.values[ir]):
                temp.append(os.path.join(seed_dir,sf))
    clustered_seed_files.append(temp)
    
for cindex,file_list in enumerate(clustered_seed_files):
    exemplar = nib.load(file_list[0])
    exemplar_dims = exemplar.get_data().shape  
    
    vect_length = np.prod(exemplar_dims)
    mask_vector = np.zeros(shape=[vect_length])
    
    for file in file_list:
        seed = nib.load(file)
        seed_data = seed.get_data()
        
        seed_vector = np.reshape(seed_data,vect_length)
        mask_vector[seed_vector==1] = 1

    mask_vol = np.reshape(mask_vector,exemplar_dims)
    mask_nifti = nib.Nifti1Image(mask_vol,affine=exemplar.affine)
    
    mask_fname = "kmeans_cluster_%s.nii.gz" % str(cindex+1)
    nib.save(mask_nifti,os.path.join(output_dir,mask_fname)) #using structural MNI space in this version