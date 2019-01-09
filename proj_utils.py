# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:29:35 2018

Common functions and data for this project

@author: ixa080020
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex,colorConverter
from collections import defaultdict
from nilearn.plotting import plot_connectome

def _get_pdir(server_path):
    """
    Gets directory for project on server
    """
    c = os.listdir(os.path.abspath(server_path))
    t = [i for i in c if "201801" in i and "Cingulate" in i][0]
    return os.path.join(server_path, t)

class Clusters(dict):
    """
    Support class for getting dendrogram cluster labels
    See get_cluster_classes
    
    Note: unreliable method - recommend using scipy.hierarchy.flcuster
        with parameters:
            criterion='maxclust'
        so that t=number of clusters
    """
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'
        return html

class proj_data():
    def __init__(self):
        self.seed_list =  [
                'PCC_inferior1_R','PCC_inferior2_R','PCC_inferior3_R',
                'PCC_inferior4_R','PCC_inferior5_R','PCC_inferior6_R',
                'PCC_inferior1_L','PCC_inferior2_L','PCC_inferior3_L',
                'PCC_inferior4_L','PCC_inferior5_L','PCC_inferior6_L',
                'PCC_superior1_R','PCC_superior2_R','PCC_superior3_R',
                'PCC_superior4_R','PCC_superior5_R','PCC_superior6_R',
                'PCC_superior7_R','PCC_superior1_L','PCC_superior2_L',
                'PCC_superior3_L','PCC_superior4_L','PCC_superior5_L',
                'PCC_superior6_L','PCC_superior7_L','ACC_inferior1_R',
                'ACC_inferior2_R','ACC_inferior3_R','ACC_inferior4_R',
                'ACC_inferior5_R','ACC_inferior6_R','ACC_inferior7_R',
                'ACC_inferior8_R','ACC_inferior9_R','ACC_inferior1_L',
                'ACC_inferior2_L','ACC_inferior3_L','ACC_inferior4_L',
                'ACC_inferior5_L','ACC_inferior6_L','ACC_inferior7_L',
                'ACC_inferior8_L','ACC_inferior9_L','ACC_superior1_R',
                'ACC_superior2_R','ACC_superior3_R','ACC_superior4_R',
                'ACC_superior5_R','ACC_superior6_R','ACC_superior7_R',
                'ACC_superior1_L','ACC_superior2_L','ACC_superior3_L',
                'ACC_superior4_L','ACC_superior5_L','ACC_superior6_L',
                'ACC_superior7_L']
        
        self.coords = [
                (5,-20,36),(5,-30,35),(5,-40,33),(5,-48,26),(5,-53,17),
                (8,-51,7),(-5,-20,36),(-5,-30,35),(-5,-40,33),(-5,-48,26),
                (-5,-53,17),(-8,-51,7),(5,-20,46),(5,-30,45),(5,-40,43),
                (5,-50,39),(5,-57,31),(5,-62,22),(5,-63,12),(-5,-20,46),
                (-5,-30,45),(-5,-40,43),(-5,-50,39),(-5,-57,31),(-5,-62,22),
                (-5,-63,12),(5,-10,37),(5,0,36),(5,10,33),(5,19,28),(5,27,21),
                (5,34,14),(5,38,6),(5,34,-4),(5,25,-10),(-5,-10,37),(-5,0,36),
                (-5,10,33),(-5,19,28),(-5,27,21),(-5,34,14),(-5,38,6),
                (-5,34,-4),(-5,25,-10),(5,-10,47),(5,2,46),(5,14,42),(5,25,36),
                (5,34,28),(5,41,21),(5,47,11),(-5,-10,47),(-5,2,46),(-5,14,42),
                (-5,25,36),(-5,34,28),(-5,41,21),(-5,47,11)]
        
        self.default_sol = np.ones(shape=[len(self.seed_list)])
        
        self.my_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
        self.network_names = [
        'DMN',
        'Sensorimotor',
        'Salience',
        'Visual',
        'Language',
        'DorsalAttention',
        'FPN',
        'Cerebellar']
        
    def get_data(self):
        """
        Returns common project data (original seed names, coords, etc.)
        """
        proj_data = {}
        proj_data['seed_list'] = self.seed_list
        proj_data['coords'] = self.coords
        proj_data['palette'] = self.my_palette
        proj_data['network_names'] = self.network_names
        return proj_data
    
def node_color_generator(color_palette,node_labels):
    """
    color_palette: list of matplotlib colors, defaults to my list
    node_labels: list of integers corresponding to cluster labels
    """        
    color_list = []
    for label in node_labels:
        color_list.append(color_palette[int(label)])
    return color_list

def get_cluster_classes(den, label='ivl'):
    """
    algorithm for getting dendrogram cluster labels
    den: dendrogram object from scipy.cluster.hierarchy.dendrogram
    
    Note: uses dendrogram from SciPy v0.14.0
    """
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l
    return cluster_classes

def unique_sol_algorithm(unlabeled_sol):
    """
    algorithm for relabeling k-means solutions
    -prep work for finding the true number of unique k-means solutions
    
    unlabeled_sol: an integer array of cluster labels
    """
    cluster_labels = pd.unique(unlabeled_sol) #returns unique labels in order of discovery!
    labeled_sol = []
    for i,x in enumerate(unlabeled_sol):
        for j,y in enumerate(cluster_labels):
            if x == y:
                labeled_sol.append(j)
    return(labeled_sol)

def convert_nii_to_gz(file,output_dir=os.getcwd()):
    """
    function for converting nifti files to compressed .gz files
    
    file: .nii file
    output_dir: default is current working directory
    """
    try:
        os.path.isdir(output_dir)
    except Exception:
        print(type(Exception))
    
    try:
        nifti = nb.load(file)
    except Exception:
        print(type(Exception))
    
    newname = "%s.gz" % file
    fpath = os.path.join(output_dir,newname)
    nb.save(nifti,fpath)

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
    Function for getting unique features from Functional Preference Profile feature dataframe output
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

def plotBrains(nodeCoords, nodeColors, fname=None):
    numc = len(nodeColors)
    adjacency_mat = np.zeros(shape=(numc, numc))
    
    brain_fig = plt.figure(figsize=[24,12])
    plot_connectome(
            adjacency_matrix=adjacency_mat,
            node_coords=nodeCoords,
            node_color=nodeColors,
            display_mode='lr',
            node_size=900,
            figure=brain_fig,
            output_file=fname,
            black_bg=False)

def plotBrains2(coords, colors, fname=None):
    #Simplified from projUtils version
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    numc = len(colors)
    adjMat = np.zeros(shape=(numc, numc))
    fig = plt.figure(figsize=[24, 12])
    plot_connectome(adjMat, coords, colors, display_mode='lr', node_size=900,
                    figure=fig, output_file=fname, black_bg=False)
    plt.show()
###############################
def _get_proj_dir_v2():
    """
	More robust version of finding the project directory
	Looks for target instead of directory name
	Just in case someone decides to rename the project folder... (._.)
    """
    pdir = os.path.abspath(os.getcwd())
    target = "this_is_proj_dir.txt"
    while not [t for t in os.listdir(pdir) if target in t]:
        pdir = os.path.abspath(os.path.dirname(pdir))
    return pdir