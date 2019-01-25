# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:22:33 2018

2nd level analysis for cingulate voxel-voxel analysis

@author: ixa080020
"""

import os
import numpy as np
import h5py
import re
import time
import random
from scipy.stats import ttest_1samp

def sleeper(tc=30):
    """Quick sleep function to prevent overlap of cluster jobs"""
    t = random.randint(1, 9)
    time.sleep(t*tc)

#--- Funtions for getting column data ---#
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

#--- Functions for doing natural sorting ---#
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

with open("secondLevPathsPrivate.txt") as f:
    paths = [k.replace("\n", "") for k in f]
    
#chunkDir = paths[0]
#dataDir = paths[1]
#outDir = paths[2]

chunkDir = r"./aal2_chunks"
dataDir = r"./cingulate_first_level_"
outDir = r"./cingulate_second_level_"

first_level_data = os.listdir(dataDir)
chunk_size = 10 #extracts "small" amounts of data at a time

aalChunks = sorted(os.listdir(chunkDir))
for aal2 in aalChunks:
    maskName = aal2.replace(".nii.gz","")
    subjects = [s for s in first_level_data if maskName in s]
    
    #Getting dataset keynames, vector size
    hf = h5py.File(os.path.join(dataDir, subjects[0]), 'r')
    keynames = list(hf.keys())
    keynames.sort(key=alphanum_key)
    brain_size = len(hf[keynames[0]][:,0])
    hf.close()

    for key in keynames:
        #sleeper()
        
        #Check for true max number of columns in current dataset
        first_vox_index = int(key.split()[-1])
        last_vox_index = int(key.split()[-3])
        if (first_vox_index - last_vox_index + 1) == 1000:
            max_columns_in_dset = 1000
        else:
            max_columns_in_dset = int(first_vox_index - last_vox_index)    
        
        #Calculate number of chunks to divide dataset into
        chunks = chunk_getter(max_columns_in_dset, chunk_size)

        for chunk in range(chunks):
            #Generate range of columns to extract volumn data
            colrange = colrange_getter(max_columns_in_dset, chunk, chunk_size)
            
            first_col = min(colrange)
            last_col = max(colrange)
            fname = "%s_cols_%d_to_%d.hdf5" % (maskName, first_col, last_col)
            fpath = os.path.join(outDir, fname)

            if os.path.isfile(fpath):
                continue
            else:
                file = h5py.File(fpath, "a")
                file.close()
            
            t_data = np.ndarray(shape=[chunk_size, brain_size])
            p_data = np.ndarray(shape=[chunk_size, brain_size])

            for v, voxel in enumerate(list(colrange)):
                conn_data = np.ndarray(shape=[len(subjects), brain_size])
                
                #--- extract connectivity data from every subject ---#
                for s,subj in enumerate(subjects):
                    subjFile = os.path.join(dataDir, subj)
                    hf = h5py.File(subjFile, 'r')
                    data = hf[key][:, int(voxel)]
                    hf.close()
                    
                    np.nan_to_num(data, copy=False)
                    data[data>=1.0] = 1 - 1e-6
                    data[data<=-1.0] = -1 + 1e-6
    
                    zData = np.arctanh(data) #fisher transform
    
                    np.nan_to_num(zData, copy=False)
                    conn_data[s, :] = zData 
                    del data, zData
    
                #--- whole brain t-test ---#    
                popmean = np.zeros(shape=[1, brain_size])
                t_brain, p_brain = ttest_1samp(conn_data, popmean, axis=0)
    
                t_data[v, :] = t_brain
                p_data[v, :] = p_brain
                del conn_data

            file = h5py.File(fpath, "a")
            file.create_dataset("tBrains", data=t_data)
            file.create_dataset("pBrains", data=p_data)
            file.close()
        
            del t_data
            del p_data
    break