# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:39:00 2018

Cingulate voxel-voxel analysis - first level [master file]
First-level and second-level scripts are tailored for ganymede/petastore
These scripts are essentially backup copies

@author: ixa080020
"""

import os
import random
import time
import h5py
import numpy as np
import pandas as pd
import nilearn
from nilearn.input_data import NiftiMasker as masker
nilearn.EXPAND_PATH_WILDCARDS = False

def sleeper(tc=30):
    """Quick sleep function to prevent overlap of cluster jobs"""
    t = random.randint(1, 9)
    time.sleep(t*tc)

def chunk_getter(maxcol, chunk_size=1000):
    """
    Calculate number of chunks to divide x_cols into
    Default chunk_size is 1000 variables per chunk
    """
    chunks = 1
    while(maxcol/chunks) > chunk_size:
        chunks += 1
    return chunks

def colrange_getter(maxcol, chunk, chunk_size=1000):
    """
    Get the range of x_cols to grab
    """
    colrange = range(chunk*chunk_size, (chunk + 1)*chunk_size)
    if max(colrange) >= maxcol:
        colrange = range(chunk*chunk_size, maxcol)
    return colrange

with open("firstLevPathsPrivate.txt") as f:
    paths = [k.replace("\n", "") for k in f]

subj_dir = paths[0]
subjs = os.listdir(subj_dir)
subj_batch = pd.DataFrame(subjs)

output_path = paths[1]
if not os.path.isdir(output_path):
    os.mkdir(output_path)

brain_mask = paths[2]
    
chunk_dir =  paths[3]
cing_files = sorted(os.listdir(chunk_dir))

for cing_file in cing_files:
    sleeper()
    cing_name = str(cing_file).replace(".nii.gz","")
    cing_mask = os.path.join(chunk_dir,cing_file)
    cing_masker = masker(cing_mask)
    
    for s in subj_batch.values:
        sleeper()
        
        #Filename structure: denoised_######.nii
        subjFile = str(s).replace("[","").replace("'","").replace("]","")
        subj_code = str(subjFile).split("_")[1].replace(".nii","")
        subj_nifti = os.path.join(subj_dir, subjFile)
        
        outfile = os.path.join(output_path, "%s_%s.hdf5" % (subj_code,cing_name))
        if os.path.isfile(outfile):
            continue
        else:
            f = h5py.File(outfile,'a')
            f.close()
            
        brain_masker = masker(brain_mask)
        brain_ts = brain_masker.fit_transform(subj_nifti)
    
        n = brain_ts.shape[1]
        s = brain_ts.shape[0]
        
        brain_mean = brain_ts.mean(0)
        brain_centered = np.subtract(brain_ts,
                                     np.reshape(np.repeat(brain_mean, brain_ts.shape[0]), brain_ts.shape, order='F'))
        brain_std = brain_ts.std(0, ddof=s-1)
        del brain_ts
        
        cing_ts = cing_masker.fit_transform(subj_nifti)
        
        chunk_size = 1000
        chunks = chunk_getter(cing_ts.shape[1], chunk_size)         
        
        f = h5py.File(outfile,'a')
        for chunk in range(chunks):
            colrange = colrange_getter(cing_ts.shape[1], chunk, chunk_size)
            chunk_name = "columns %07d to %07d" % (min(colrange),max(colrange))
            cing_chunk = cing_ts[:, colrange]
            
            m = cing_chunk.shape[1]
            cing_chunk_mean = cing_chunk.mean(0)
            cing_chunk_centered = np.subtract(cing_chunk,
                                              np.reshape(np.repeat(cing_chunk_mean, cing_chunk.shape[0]), cing_chunk.shape, order='F'))
            cing_chunk_std = cing_chunk.std(0, ddof=s-1)
            del cing_chunk
            
            cov = np.dot(cing_chunk_centered.T, brain_centered)
            r = cov/np.dot(cing_chunk_std[:, np.newaxis], brain_std[np.newaxis, :])
            del cov
            
            f.create_dataset(chunk_name,
                             shape=[brain_centered.shape[1], cing_chunk_centered.shape[1]],
                             dtype='f',
                             compression='lzf',
                             data=r)
            del r
        f.close()
           
        del cing_ts  