# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:17 2019
"""

import os
import h5py
import datetime
import numpy as np
from copy import deepcopy
from mne.stats import fdr_correction
from nilearn.masking import unmask, apply_mask
from nistats.thresholding import map_threshold

def clean_p_vector(p_vector):
    #Convert nan to .999999
    correction = 1 - 1e-6
    cleaned_p_vector = [p if not np.isnan(p) else correction for p in p_vector]
    return np.asarray(cleaned_p_vector)

mni_mask = 'mni152T1mask2mm.nii'
data_dir = r'./cingulate_second_level_'
out_dir = r'./cingulate_second_level_thresh_'

aal2_dir = r'./aal2_chunks'
aal2_masks = sorted(os.listdir(aal2_dir))
aal2_masknames = [a.replace('.nii.gz', '') for a in aal2_masks]

print('Starting at %s' % str(datetime.datetime.now()))
for aal2 in aal2_masknames:
    fname = aal2 + '_thresh.hdf5'
    outfile = h5py.File(os.path.join(out_dir, fname), 'a')
    
    for file in sorted(os.listdir(data_dir)):

        if aal2 in file:
            dataset_name = file.replace('.hdf5', '')
            dset_name1 = dataset_name + '_data'
            dset_name2 = dataset_name + '_masks'
            
            #Quick check to see if data is already in the hdf5 output file
            key_check = list(outfile.keys())
            if dset_name1 in key_check or dset_name2 in key_check:
                continue
            
            #Extract unthresholded second level data
            data_path = os.path.join(data_dir, file)
            hf = h5py.File(data_path, 'r')
            dset_names = sorted(list(hf.keys()))
            
            p_data = hf[dset_names[0]][:, :]
            t_data = hf[dset_names[1]][:, :]
            hf.close()
            
            #Initialize output arrays
            thresh_data = np.ndarray(shape=t_data.shape)
            thresh_masks = np.ndarray(shape=t_data.shape)

            #Threshold voxel->brain second level results, one voxel at a time
            for v in range(t_data.shape[0]):
                t_vect = t_data[v, :]
                p_vect = p_data[v, :]
                
                #Get rid of nans in p values
                cleaned_p_vect = clean_p_vector(p_vect)
                
                #Fdr correction
                _, fdr = fdr_correction(cleaned_p_vect)
                p_fdr = np.asarray(fdr)
                p_fdr[p_fdr>.05] = 0
                
                #Cluster correction
                brain_3d = unmask(p_fdr, mni_mask)
                thresh = map_threshold(brain_3d,
                                       height_control=None,
                                       cluster_threshold=200)
                
                #Flatten brain back into to vector
                thresh_vect = apply_mask(thresh[0], mni_mask)
                
                #Threshold t_brain data, create (flattened) mask
                t_vect[thresh_vect==0] = 0
                t_mask = deepcopy(t_vect)
                t_mask[t_mask!=0] = 1
                
                thresh_data[v, :] = t_vect
                thresh_masks[v, :] = t_mask
            
            #Append thresholded data to hdf5 output files
            outfile.create_dataset(name=dset_name1,
                                    data=thresh_data,
                                    compression='gzip')
            outfile.create_dataset(name=dset_name2,
                                    data=thresh_masks,
                                    compression='gzip')
            
            c = str(datetime.datetime.now())
            print(dataset_name + ' completed at ' + c)
            
    outfile.close()