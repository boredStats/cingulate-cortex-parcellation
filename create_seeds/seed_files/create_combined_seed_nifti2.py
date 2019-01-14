# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:24:30 2018

Create one big seed nifti mask
"""

import os
import nibabel as nib
import numpy as np

nifti_dir = os.path.join(os.getcwd(),"nifti")

img_files = os.listdir(nifti_dir)

seed_example = nib.load(os.path.join(nifti_dir,img_files[0]))
seed_example_dims = seed_example.get_data().shape
seed_example_affine = seed_example.affine


vect_length = np.prod(seed_example_dims)
mask_vector = np.zeros(shape=[vect_length])

for file in img_files:
    seed = nib.load(os.path.join(nifti_dir,file))
    seed_data = seed.get_data()
    
    seed_vector = np.reshape(seed_data,vect_length)
    
    mask_vector[seed_vector==1] = 1

mask_vol = np.reshape(mask_vector,seed_example_dims)
mask_nifti = nib.Nifti1Image(mask_vol,affine=seed_example_affine)

mask_fname = "all_seeds.nii.gz" 
nib.save(mask_nifti,os.path.join(os.getcwd(),mask_fname))

"""
Create one big seed nifti mask, with each seed having a different index
"""
vect_length = np.prod(seed_example_dims)
mask_vector = np.zeros(shape=[vect_length])

i = 1
for file in img_files:
    seed = nib.load(os.path.join(nifti_dir,file))
    seed_data = seed.get_data()
    
    seed_vector = np.reshape(seed_data,vect_length)
    
    mask_vector[seed_vector==1] = i
    i += 1

mask_vol = np.reshape(mask_vector,seed_example_dims)
mask_nifti = nib.Nifti1Image(mask_vol,affine=seed_example_affine)

mask_fname = "all_seeds_indexed.nii.gz" 
nib.save(mask_nifti,os.path.join(os.getcwd(),mask_fname))

"""
Create ACC nifti 
"""
acc_rois = [os.path.join(nifti_dir, r) for r in os.listdir(nifti_dir) if "ACC" in r]
vect_length = np.prod(seed_example_dims)
mask_vector = np.zeros(shape=[vect_length])

for file in acc_rois:
    seed = nib.load(file)
    seed_data = seed.get_data()
    
    seed_vector = np.reshape(seed_data,vect_length)
    
    mask_vector[seed_vector==1] = 1
mask_vol = np.reshape(mask_vector,seed_example_dims)
mask_nifti = nib.Nifti1Image(mask_vol,affine=seed_example_affine)

mask_fname = "acc_seeds.nii.gz" 
nib.save(mask_nifti, mask_fname)
"""
Create PCC nifti 
"""
pcc_rois = [os.path.join(nifti_dir, r) for r in os.listdir(nifti_dir) if "PCC" in r]
vect_length = np.prod(seed_example_dims)
mask_vector = np.zeros(shape=[vect_length])

for file in pcc_rois:
    seed = nib.load(file)
    seed_data = seed.get_data()
    
    seed_vector = np.reshape(seed_data,vect_length)
    
    mask_vector[seed_vector==1] = 1
mask_vol = np.reshape(mask_vector,seed_example_dims)
mask_nifti = nib.Nifti1Image(mask_vol,affine=seed_example_affine)

mask_fname = "pcc_seeds.nii.gz" 
nib.save(mask_nifti, mask_fname)

