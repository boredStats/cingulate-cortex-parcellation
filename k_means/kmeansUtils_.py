# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:48:03 2018

Utilities for kmeans analysis - written for custom kmeans scripts - older version
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans(data, nk, niters=256):
    sil_scores_bil = np.zeros(shape=[niters,nk-2])
    clust_labels_bil = []
    results = {}
    for i in range(niters):
        start_time = time.time()
        print("Iteration: %d out of %d" % (i+1,niters))
        cluster_labels = []
        for k_index,k in enumerate(range(2,nk)):
            kmeans = KMeans(
                    n_clusters=k,
                    verbose=0,
                    init='k-means++',
                    n_init=100)
            k_means_calc = kmeans.fit_predict(data)
            cluster_labels.append(k_means_calc)
            sil_score = silhouette_score(data, k_means_calc)
            sil_scores_bil[i,k_index] = sil_score
        clust_labels_bil.append(cluster_labels)
        end_time = time.time()
        calc_time = (end_time-start_time)/60
        print("Calculation time: %.3f minutes" % calc_time)
        
    results['silScores'] = sil_scores_bil
    results['clustLabels'] = clust_labels_bil
    return results

def plotSilScores(km_data, fname=None):
    
    silScores = km_data['silScores']
    
    temp = km_data['silScores']
    silScores = temp[:,:28]
    
    sil_var = np.var(silScores*1000,axis=0)
    sil_pre = 1/sil_var
    sil_pre[sil_pre>1] = 2
    
    sil_scores_vect = np.ndarray.flatten(silScores)
    max_sil = np.max(sil_scores_vect)
    sil_scores_vect = sil_scores_vect/max_sil
    
    xcoords = np.arange(silScores.shape[1])
    sil_xcoords = np.tile(xcoords,reps=silScores.shape[0])
    
    mean_sil_scores = np.mean(silScores,axis=0)
    best_k = np.argmax(mean_sil_scores)+2
    
    fig,ax1 = plt.subplots()
    
    myBlue = 'tab:blue'
    myRed = 'tab:red'
    
    ax1.scatter(x=sil_xcoords,y=sil_scores_vect,alpha=.02,color=myBlue,s=50)
    #ax1.scatter(x=[1,3],y=[sil_scores_vect[1],sil_scores_vect[3]],alpha=1,color=myBlue,s=200,marker="*")
    
    plt.xticks(xcoords,2+xcoords)
    plt.tick_params(axis='both',top=False,labeltop=False,labelbottom=True,bottom=True,labelsize=12)
    plt.xlabel(r"Number of clusters $(k)$",fontsize=18)
    ax1.set_ylabel("Silhouette score (normalized)",fontsize=18,color=myBlue)
    ax1.tick_params('y', colors=myBlue)
    ax1.set_ylim([.725,1.01])
    
    ax2 = ax1.twinx()
    ax2.plot(xcoords,sil_pre,'k:',linewidth=1,color=myRed)
    ax2.set_ylabel(r"Precision $(\frac{1}{\sigma^2})$",fontsize=18,color=myRed)
    ax2.tick_params('y', colors=myRed)
    ax2.set_ylim([0,1])
    
    if fname is not None:
        fig.set_size_inches([12,9])
        fig.savefig(fname, bbox_inches='tight', dpi=600)
    
    return best_k

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

def extractSol(km_data, best_k):
    clustLabels = km_data['clustLabels']
    n_iter = len(clustLabels)
    solution_list = np.empty(shape=[58,len(clustLabels)])
    for i in range(n_iter):
        # subtracting by 2 because k means calculations start at k = 2
        cluster_solution = clustLabels[i][best_k-2]
        solution_list[:,i] = cluster_solution
    
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
    bilateral_consistency = np.zeros(shape=[58,2])        
    for x in range(58):
        cluster = np.argmax(sol_counter[x,:])
        n_occurences = np.ndarray.max(sol_counter[x,:])
        percentage = n_occurences/n_iter
        bilateral_consistency[x,0] = cluster
        bilateral_consistency[x,1] = percentage

    return bilateral_consistency