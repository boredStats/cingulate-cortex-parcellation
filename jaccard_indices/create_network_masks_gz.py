# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:06:08 2018

Create masks from Conn toolbox's network.nii file

1-4 DMN
5-7 Sensorimotor
8-11 Visual
12-18 Salience
19-22 Dorsal Attention
23-26 Frontoparietal
27-30 Language
31-32 Cerebellar
"""
import os
import pandas as pd
from nibabel import load,Nifti1Image,save
from copy import deepcopy

def mask_creator(empty_mask,map_list,dims):
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if any(map_list[x,y,z,:]):
                    empty_mask[x,y,z] = 1
                else:
                    empty_mask[x,y,z] = 0
    return empty_mask

odir = r"./network_masks"
if not os.path.isdir(odir):
    os.mkdir(odir)

networks = pd.read_table("networks.txt", header=None)
raw_names = []
for n in networks.values:
    raw_name = n[0].split()[0]
    network_name = raw_name.split('.')[0]
    raw_names.append(network_name)
names_df = pd.Series(raw_names)
network_list = list(pd.Series.unique(names_df))

with open("network_names.txt", "w") as f:
    for n in network_list:
        f.write("%s\n" % n)

range_list = [(0, 3), (4, 6), (7, 10), (11, 17), (18, 21), (22, 25), (26, 29), (30, 31)]

network_data = load("networks.nii.gz")
network_ROIs = network_data.get_data()
temp_mask = deepcopy(network_ROIs[:,:,:,0])
temp_dims = temp_mask.shape

for i, n in enumerate(network_list):
    x = range_list[i][0]
    y = range_list[i][1]
    rois = network_ROIs[:, :, :, x:y]
    mask = mask_creator(temp_mask, rois, temp_dims)
    img = Nifti1Image(mask, network_data.affine, network_data.header)
    f = "%s_mask.nii.gz" % n
    save(img, os.path.join(odir, f))
