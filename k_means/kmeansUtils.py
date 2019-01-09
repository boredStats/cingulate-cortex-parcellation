# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:48:03 2018

Utilities for kmeans analysis - written for custom kmeans scripts
"""
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans(data, k, niters=256):
    typeC = .66
    if issparse(data):
        typeC = 2.66
    shapeC = (data.shape[0]*data.shape[1]) / 50**2
    print("Estimated runtime: %.2f minutes" % (niters* typeC * shapeC))
    start_time = time.time()
    
    #Output structures
    sils = np.zeros(shape=[niters, k-2])
    clust = {}
    
    ilabs = []
    for n in range(niters):
        ilab = "Iteration_%03d" % (n+1)
        ilabs.append(ilab)
        
        print("Iteration: %d out of %d..." % (n+1, niters))
        cluster_labels = {}
        for i, ki in enumerate(range(2, k)):
            klab = "k=%d" % ki
            kmeans = KMeans(n_clusters=ki, init='k-means++', n_init=100)
            k_means_calc = kmeans.fit_predict(data)
            cluster_labels[klab] = k_means_calc
            
            sil_score = silhouette_score(data, k_means_calc)
            sils[n, i] = sil_score

        clust[ilab] = cluster_labels
    sil_df = pd.DataFrame(sils, index=ilabs, columns=np.arange(2, k))
    
    end_time = time.time()
    calc_time = (end_time-start_time)/60
    print("Calculation time: %.2f minutes" % calc_time)
    return clust, sil_df

def bestSilScores(sil_scores, max_k=None, fname=None):   
    if not max_k:
        max_k = int(sil_scores.values.shape[1])
    else:
        max_k = int(max_k)     
    scores = sil_scores.values[:,:max_k]
    
    scale_factor = 1000
    sil_variance = np.var(scores*scale_factor, axis=0)
    sil_variance[sil_variance==0] = 1e-9 #For precision calculuation
    sil_precision = 1 / sil_variance
    sil_precision[sil_precision > 1] = 2
    
    sil_vector = np.ndarray.flatten(scores)
    max_sil = np.max(sil_vector)
    min_sil = np.min(sil_vector)
    
    sil_vector = sil_vector/max_sil
    min_sil = np.min(sil_vector) - .05
    
    xpos = np.arange(scores.shape[1])
    xcoord = np.tile(xpos, reps=scores.shape[0]) #match each tick with vector index
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.clf()
    fig,ax1 = plt.subplots()
    
    #Figure prep
    blu = 'tab:blue'
    red = 'tab:red'
    alpha = .3 #originally .2
    if scores.shape[0] < 50:
        alpha = .6

    ax1.scatter(x=xcoord, y=sil_vector, alpha=alpha, color=blu, s=50)
    ax1.set_ylabel("Silhouette score (normalized)", fontsize=18, color=blu)
    ax1.tick_params('y', colors=blu)
    ax1.set_ylim([min_sil, 1.01])
    
    plt.xticks(xpos, 2+xpos)
    plt.tick_params(axis='both', top=False, labeltop=False, labelbottom=True,
                    bottom=True, labelsize=12)
    plt.xlabel(r"Number of clusters $(k)$", fontsize=18)
    
    ax2 = ax1.twinx()
    ax2.plot(xpos, sil_precision, 'k:', linewidth=1, color=red)
    ax2.set_ylabel(r"Precision $(\frac{1}{\sigma^2})$", fontsize=18, color=red)
    ax2.tick_params('y', colors=red)
    ax2.set_ylim([0, 1])
    plt.show()
    if fname:
        fig.set_size_inches([9, 6])
        fig.savefig(fname, bbox_inches='tight', dpi=600)
    
    mean_iteration_scores = np.mean(scores, axis=0)
    best_k = np.argmax(mean_iteration_scores) + 2
    return best_k

def bestSolution(km_data, best_k, obs_names=None):
    
    def unique_sol_algorithm(unlabeled_sol):
        """
        algorithm for relabeling k-means solutions
        -prep work for finding the true number of unique k-means solutions
        
        unlabeled_sol: an integer array of cluster labels
        Note: pandas unique function returns unique labels in order of discovery
        """
        cluster_labels = pd.unique(unlabeled_sol) 
        labeled_sol = []
        for i,x in enumerate(unlabeled_sol):
            for j,y in enumerate(cluster_labels):
                if x == y:
                    labeled_sol.append(j)
        return(labeled_sol)
        
    n_iter = len(km_data)
    solution_list = np.empty(shape=[58,len(km_data)])
    k_key = "k=%d" % (best_k)
    for index, big_key in enumerate(km_data):
        # subtracting by 2 because k means calculations start at k = 2
        cluster_solution = km_data[big_key][k_key]
        solution_list[:, index] = cluster_solution
    
    # Getting true solutions 
    unique_sols,unique_counts = np.unique(solution_list,axis=1,return_counts=True)
    labeled_sols = np.empty(shape=unique_sols.shape)
    for a in range(len(unique_counts)):
        sol = unique_sols[:,a]
        true_sol = unique_sol_algorithm(sol)
        labeled_sols[:,a] = true_sol
    
    # Counting true unique solutions
    sol_counter = np.zeros(shape=[58,best_k])  
    for x in range(58):
        seed_labels = labeled_sols[x,:]
        for y in range(len(seed_labels)):
            for z in range(best_k):
                if seed_labels[y] == z:
                    sol_counter[x,z] = np.add(sol_counter[x,z],unique_counts[y])
                    
    # Getting consistency of the labelling schemes
    sol_consistency = np.zeros(shape=[58,2])        
    for x in range(58):
        cluster = np.argmax(sol_counter[x,:])
        n_occurences = np.ndarray.max(sol_counter[x,:])
        percentage = n_occurences/n_iter
        sol_consistency[x,0] = cluster
        sol_consistency[x,1] = percentage
    
    colnames = ['Best cluster assignment',
                'Percentage of iterations assigned to cluster']
    sol_df = pd.DataFrame(sol_consistency, index=obs_names, columns=colnames)
    return sol_df