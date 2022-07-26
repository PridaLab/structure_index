import sys
sys.path.append('/media/enrique/Disk1/Proyectos/UnsupervisedRippleClassification/Code/Python')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, linalg
import pandas as pd
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap


def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nnDist = np.sum(D < np.nanpercentile(D,1), axis=1) - 1
    noiseIdx = np.where(nnDist < np.percentile(nnDist, 20))[0]
    return noiseIdx


def compute_pointCloudsOverlap(cloud1, cloud2, k, distance_metric):
    #Stack both clouds
    cloud_all = np.vstack((cloud1, cloud2))
    #Create cloud label
    cloud_label = np.hstack((np.ones(cloud1.shape[0]), np.ones(cloud2.shape[0])*2))
    #Compute k neighbours graph wieghted by cloud label
    if distance_metric == 'euclidean':
        connectivity = kneighbors_graph(X=cloud_all, n_neighbors=k, mode='connectivity', include_self=False).toarray() * cloud_label
    elif distance_metric == 'geodesic':
        model_iso = Isomap(n_components = 1)
        emb = model_iso.fit_transform(cloud_all)
        dist_mat = model_iso.dist_matrix_
        knn_distance_based = (NearestNeighbors(n_neighbors=3, metric="precomputed").fit(dist_mat))
        connectivity = knn_distance_based.kneighbors_graph(mode='connectivity').toarray()  * cloud_label
    #Compute the degree of each point of cloud 1
    degree = np.sum(connectivity, axis=1)[cloud_label==1]
    #Compute overlap percentage
    overlap = (np.sum(degree > k) / degree.shape[0])*100
    return overlap


def computeClusterIndex_V4(emb, label, nBins, dimNames, overlapThreshold=0.5, distance_metric='euclidean', **kwargs):
    #Preprocess data 
    emb = emb[dimNames].to_numpy()
    # for d in range(emb.shape[1]):
        # emb[:,d] = (emb[:,d] - np.nanmean(emb[:,d])) / np.nanstd(emb[:,d])
    #Delete nan values from label and emb
    emb = np.delete(emb, np.where(np.isnan(label))[0], axis=0)
    label = np.delete(label, np.where(np.isnan(label))[0], axis=0)
    #If there is a predefined max or min delete all points out of bounds
    if 'vmin' in kwargs:
        #emb = np.delete(emb, np.where(label<kwargs['vmin'])[0], axis=0)
        label[np.where(label<kwargs['vmin'])[0]] = kwargs['vmin']
        #label = np.delete(label, np.where(label<kwargs['vmin'])[0], axis=0)
    if 'vmax' in kwargs:
        #emb = np.delete(emb, np.where(label>kwargs['vmax'])[0], axis=0)
        label[np.where(label>kwargs['vmax'])[0]] = kwargs['vmax']
        #label = np.delete(label, np.where(label>kwargs['vmax'])[0], axis=0)
    #Create the bin edges
    if 'vmin' in kwargs and 'vmax' in kwargs:
        binSize = (kwargs['vmax'] - kwargs['vmin']) / nBins
        binEdges = np.column_stack((np.linspace(kwargs['vmin'],kwargs['vmin']+binSize*(nBins-1),nBins),
                                    np.linspace(kwargs['vmin'],kwargs['vmin']+binSize*(nBins-1),nBins)+binSize))
    else:
        binSize = (np.max(label) - np.min(label)) / nBins
        binEdges = np.column_stack((np.linspace(np.min(label),np.min(label)+binSize*(nBins-1),nBins),
                                    np.linspace(np.min(label),np.min(label)+binSize*(nBins-1),nBins)+binSize))

    #Create binLabel
    binLabel = np.zeros(label.shape)
    for b in range(nBins-1):
        binLabel[np.logical_and(label >= binEdges[b,0], label<binEdges[b,1])] = 1 + int(np.max(binLabel))
    binLabel[np.logical_and(label >= binEdges[nBins-1,0], label<=binEdges[nBins-1,1])] = 1 + int(np.max(binLabel))

    #Clean outliers from each cluster if specified in kwargs
    if 'filterNoise' in kwargs and kwargs['filterNoise']:
        for l in np.unique(binLabel):
            noiseIdx = filter_noisy_outliers(emb[binLabel==l,:])
            noiseIdx = np.where(binLabel==l)[0][noiseIdx]
            binLabel[noiseIdx] = 0


    #Discard outlier clusters (nPoints < 1%)
    #Compute number of points in each cluster
    nPoints = np.array([np.sum(binLabel==value) for value in np.unique(binLabel)])
    #Get the clusters that meet criteria and delete them
    delLabels = np.where(nPoints < label.size*1/100)[0]
    delLabels = np.where(nPoints < 9)[0] #n_neighbor * 3
    #Delete outlier clusters
    for delInd in delLabels:
        binLabel[binLabel==delInd+1] = 0
    #Renumber bin labels from 1 to n clusters
    uniqueVal = np.unique(binLabel)
    if 0 in np.unique(binLabel):
        for idx in range(1,len(uniqueVal)):
            binLabel[binLabel==uniqueVal[idx]]= idx

    #Compute the cluster index
    #Compute overlap between clusters pairwise
    overlapMat = np.zeros((np.sum(np.unique(binLabel) > 0), np.sum(np.unique(binLabel) > 0)))
    for ii in range(overlapMat.shape[0]):
        for jj in range(overlapMat.shape[1]):
            if ii != jj:
                overlap = compute_pointCloudsOverlap(emb[binLabel==ii+1], emb[binLabel==jj+1], 3, distance_metric)
                overlapMat[ii,jj] = overlap/100
    #Symetrize overlap matrix
    overlapMat = (overlapMat + overlapMat.T) / 2
    clusterIndex = 1 - np.mean(np.sum(1*(overlapMat>=overlapThreshold), axis=0))/(overlapMat.shape[0]-1)

    return clusterIndex, binLabel, overlapMat