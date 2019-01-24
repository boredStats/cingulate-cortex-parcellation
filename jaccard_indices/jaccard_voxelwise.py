# -*- coding: utf-8 -*-
"""
Calculate jaccard indices voxel-voxel connectivity -> ICA networks

Build Jaccard matrices without overwriting old work
Created on Wed Jan 23 16:31:26 2019
"""

import os
import h5py
import datetime
import numpy as np
import pickle as pkl
from nilearn import image, masking
from sklearn.metrics import jaccard_similarity_score as jsc

with open(r"../pData.pkl", 'rb') as f:
    p_data = pkl.load(f)    
network_names = p_data['network_names']
server_path = p_data['server_path']

#Setting path to voxelwise data
v_path = os.path.join(server_path, 'voxelwise')
thresh_dir = os.path.join(v_path, 'cingulate_second_level_thresh_')

#Getting list of network mask files
net_dir = r'./network_masks'
net_masks = [os.path.join(net_dir, n) for n in os.listdir(net_dir)]

#Resampling network masks to match voxelwise thresh mask dimensions
mni = r'./mni152T1mask2mm.nii'
net_resamp = image.resample_to_img(net_masks, mni)       
net_vects = masking.apply_mask(net_resamp, mni) #vectorizing network masks

#Read/write to hdf5 file
j_path = os.path.join(server_path, 'jaccard_indices')
out_file = h5py.File(os.path.join(j_path, 'jaccard_voxel_to_network.hdf5'))
out_keys = list(out_file.keys())

thresh_files = sorted(os.listdir(thresh_dir))
for file in thresh_files:
    print('Trying %s: %s' % (file, str(datetime.datetime.now())))
    data_file = h5py.File(os.path.join(thresh_dir, file), 'r')
    data_keys = list(data_file.keys())
    
    mask_keys = [d for d in data_keys if '_mask' in d]
    for mk in mask_keys:
        if mk in out_keys: 
            continue #check if t-test data has already been jaccard'ed
       
        print('Running %s: %s' % (mk, str(datetime.datetime.now())))
        data = data_file[mk]
        jaccard_mat = np.ndarray(shape=[data.shape[0], len(net_masks)])
        
        for voxel_index in range(data.shape[0]):
            mask_vect = data[voxel_index, :]
            
            for network_index in range(net_vects.shape[0]):
                net_vect = net_vects[network_index, :]
                net_vect[net_vect < 1] = -1 #Zero-value voxels are dissimilar
                
                j_score = jsc(mask_vect.astype(int), net_vect.astype(int))
                jaccard_mat[voxel_index, network_index] = j_score
        
        out_file.create_dataset(name=mk, data=jaccard_mat)
        
out_file.close()
print('Finished at %s' % str(datetime.datetime.now()))