import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, linalg
import pandas as pd
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
import warnings
import copy

overlap_options = ['one_third','continuity']
graph_options = ['binary', 'weighted']

def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,1), axis=1) - 1
    noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    return noiseIdx


def compute_pointCloudsOverlap(cloud1, cloud2, k, distance_metric, overlap_method):
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

    if overlap_method == 'continuity': #total fraction of neighbours that belong to the other cloud
        overlap_1_2 = np.sum(connectivity[cloud_label==1,:]==2)/(cloud1.shape[0]*k)
        overlap_2_1 = np.sum(connectivity[cloud_label==2,:]==1)/(cloud2.shape[0]*k)
    elif overlap_method == 'one_third':
        #Compute overlap threshold for each individual point
        overlap_th = k/3
        degree_1 = np.sum(connectivity==2, axis=1)[cloud_label==1]
        overlap_1_2 = (np.sum(degree_1 >= overlap_th) / degree_1.shape[0])

        degree_2 = np.sum(connectivity==1, axis=1)[cloud_label==2]
        overlap_2_1 = (np.sum(degree_2 >= overlap_th) / degree_2.shape[0])

    return overlap_1_2, overlap_2_1


def compute_structure_index(data, label, n_bins=10, dims=None, distance_metric='euclidean', **kwargs):
    #TODO:
        #distance_metric cosyne?
        #re-evaluate which arguments put outside whichones in kwargs
        #write function definition
        #do shuffling to compute p-value (1000 times? put it also as kwarg)
        #re-think compute_pointCloudsOverlap function name
        #check all imports are being used
        #include plot-function
        #in case of discrete label, define discrete bins (done)
        #check n-neighbours vs num points per bin

    #0.Quick assesment of input's validity
    #i) data input
    assert isinstance(data, np.ndarray), f"Input 'data' must be a numpy ndarray but it was a {type(data)}"
    assert data.ndim==2, "Input 'data' must be a 2D numpy ndarray with shape (n,m) where n is the number of samples"+\
                            " and m the number of dimensions."
    #ii) label input
    assert isinstance(label, np.ndarray), f"Input 'label' must be a numpy ndarray but it was a {type(label)}"
    if label.ndim==2 and label.shape[1] == 1: #if label 2D with one column transform into column vector
            label = label[:,0]
    assert label.ndim==1, "label must be a 1D array (or 2D with only one column)."
    #iii) n_bins input
    assert isinstance(n_bins, int) or isinstance(n_bins, np.integer), "Input 'n_bins' must be an integer larger than 1."
    assert n_bins>1, "Input 'n_bins' must be an integer larger than 1."
    #iv) dims input
    if isinstance(dims, type(None)): #if dims is None, then take all dimensions
        dims = list(range(data.shape[1]))
    assert isinstance(dims, list), "Input 'dims' must be a list with the dimensions to keep or None to keep all."
    #v) overlap_method input
    if 'overlap_method' in kwargs:
        overlap_method = kwargs['overlap_method']
        assert overlap_method in overlap_options, f"Invalid input 'overlap_method'. Valid graph options are {overlap_options}."
    else:
        overlap_method = 'one_third'
    #vi) graph_type input
    if 'graph_type' in kwargs:
        graph_type = kwargs['graph_type']
        assert graph_type in graph_options, f"Invalid input 'graph_type'. Valid graph options are {graph_options}."
    else:
        graph_type = 'binary'
    #vii) overlap_threshold input
    if graph_type == 'binary':
        if 'overlap_threshold' in kwargs:
            overlap_threshold = kwargs['overlap_threshold']
            assert overlap_threshold>0 and overlap_threshold<=1, "Input 'overlap_threshold' must belong to interval (0,1]."
        else:
            overlap_threshold = 0.5
    elif graph_type == 'weighted' and 'overlap_threshold' in kwargs:
         warnings.warn(f"Input 'graph_type' is not 'binary' ('{graph_type}') but input 'overlap_threshold' provided. "
                        "It will be ignored.")
    #viii) n_neighbors input
    if 'n_neighbors' in kwargs:
        n_neighbors = kwargs['n_neighbors']
        assert isinstance(n_neighbors,int), "Input 'n_neighbors' must be an integer larger than 2."
        assert n_neighbors>2, "Input 'n_neighbors' must be an integer larger than 2."
    else:
        n_neighbors = 3
    #ix) discrete_bin_label input
    if 'dicrete_bin_label' in kwargs:
        discrete_bin_label = kwargs['dicrete_bin_label']
        assert isinstance(discrete_bin_label, bool), "Input 'discrete_bin_label' must be a boolean."
    else:
        discrete_bin_label = False
    #x) num_shuffles input
    if 'num_shuffles' in kwargs:
        num_shuffles = kwargs['num_shuffles']
    else:
        num_shuffles = 100

    #1.Preprocess data
    data = data[:,dims] #keep only desired dims
    if data.ndim == 1: #if left with 1 dim, keep the 2D shape
        data = data.reshape(-1,1)

    #2.Delete nan values from label and data
    data_nans = np.any(np.isnan(data), axis = 1)
    label_nans = np.isnan(label)
    delete_nans = np.where(data_nans+label_nans)[0]

    data = np.delete(data,delete_nans, axis=0)
    label = np.delete(label,delete_nans, axis=0)

    #3.Binarize label
    if discrete_bin_label: #if discrete label
        bin_label = np.zeros(label.shape)
        unique_label = np.unique(label)
        for b in unqiue_label:
            bin_label[label == b] = 1 + int(np.max(bin_label))

    else: #if continuous label
        #i) Check bin-num vs num unique label
        num_unique_label = len(np.unique(label))
        if n_bins>num_unique_label:
             warnings.warn(f"Input 'label' has less unique values ({num_unique_label}) than specified 'n_bins' "
                            f"({n_bins}). Changing 'n_bins' to {num_unique_label}.")
             n_bins = num_unique_label

        #ii) Create bin edges of bin-groups
        if 'min_label' in kwargs:
            min_label = kwargs['min_label']
            label[np.where(label<min_label)[0]] = min_label
        else:
            min_label = np.min(label)

        if 'max_label' in kwargs:
            max_label = kwargs['max_label']
            label[np.where(label>max_label)[0]] = max_label
        else:
            max_label = np.max(label)

        bin_size = (max_label - min_label) / n_bins
        bin_edges = np.column_stack((np.linspace(min_label,min_label+bin_size*(n_bins-1),n_bins),
                                    np.linspace(min_label,min_label+bin_size*(n_bins-1),n_bins)+bin_size))

        #iii) Create bin_label
        bin_label = np.zeros(label.shape)
        for b in range(n_bins-1):
            bin_label[np.logical_and(label >= bin_edges[b,0], label<bin_edges[b,1])] = 1 + int(np.max(bin_label))
        bin_label[np.logical_and(label >= bin_edges[n_bins-1,0], label<=bin_edges[n_bins-1,1])] = 1 + int(np.max(bin_label))

    #6. Clean outliers from each bin-groups if specified in kwargs
    if 'filter_noise' in kwargs and kwargs['filter_noise']:
        for l in np.unique(bin_label):
            noise_idx = filter_noisy_outliers(data[bin_label==l,:])
            noise_idx = np.where(bin_label==l)[0][noise_idx]
            bin_label[noise_idx] = 0

    #7. Discard outlier bin-groups (n_points < n_neighbors)
    #i) Compute number of points in each bin-group
    n_points = np.array([np.sum(bin_label==value) for value in np.unique(bin_label)])
    #ii) Get the bin-groups that meet criteria and delete them
    del_labels = np.where(n_points < n_neighbors)[0]
    #iii) delete outlier bin-groups
    for del_idx in del_labels:
        bin_label[bin_label==del_idx+1] = 0
    #iv) renumber bin labels from 1 to n bin-groups
    unique_bin_label = np.unique(bin_label)
    if 0 in np.unique(bin_label):
        for idx in range(1,len(unique_bin_label)):
            bin_label[bin_label==unique_bin_label[idx]]= idx

    #8. Compute the structure index
    #i) compute overlap between bin-groups pairwise
    overlap_mat = np.zeros((np.sum(np.unique(bin_label) > 0), np.sum(np.unique(bin_label) > 0)))
    for ii in range(overlap_mat.shape[0]):
        for jj in range(ii+1, overlap_mat.shape[1]):
            overlap_1_2, overlap_2_1 = compute_pointCloudsOverlap(data[bin_label==ii+1], data[bin_label==jj+1], n_neighbors, distance_metric,overlap_method)
            overlap_mat[ii,jj] = overlap_1_2
            overlap_mat[jj,ii] = overlap_2_1
    #ii) symetrize overlap matrix
    overlap_mat = (overlap_mat + overlap_mat.T) / 2
    #iii) computed structure_index
    if graph_type=='binary':
        structure_index = 1 - np.mean(np.sum(1*(overlap_mat>=overlap_threshold), axis=0))/(overlap_mat.shape[0]-1)
    elif graph_type=='weighted':
        structure_index = 1 - np.mean(np.sum(overlap_mat, axis=0))/(overlap_mat.shape[0]-1)

    #9. Shuffling
    verbose = True
    if verbose:
        print('Computing shuffling: X/X', sep='', end = '') 
        pre_del = '\b'*3
    shuf_structure_index = np.zeros((num_shuffles,))
    shuf_overlap_mat = np.zeros((overlap_mat.shape))
    for s_idx in range(num_shuffles):
        if verbose:
            print(f"{pre_del}{s_idx}/{num_shuffles}", sep='', end = '')
            pre_del = '\b'*(len(str(s_idx))+len(str(num_shuffles))+1)
        shuf_bin_label = copy.deepcopy(bin_label)
        np.random.shuffle(shuf_bin_label)
        shuf_overlap_mat *= 0
        for ii in range(shuf_overlap_mat.shape[0]):
            for jj in range(ii+1, shuf_overlap_mat.shape[1]):
                overlap_1_2, overlap_2_1 = compute_pointCloudsOverlap(data[shuf_bin_label==ii+1], data[shuf_bin_label==jj+1], n_neighbors, distance_metric,overlap_method)
                shuf_overlap_mat[ii,jj] = overlap_1_2
                shuf_overlap_mat[jj,ii] = overlap_2_1
        #ii) symetrize overlap matrix
        shuf_overlap_mat = (shuf_overlap_mat + shuf_overlap_mat.T) / 2
        #iii) computed structure_index
        if graph_type=='binary':
            shuf_structure_index[s_idx] = 1 - np.mean(np.sum(1*(overlap_mat>=overlap_threshold), axis=0))/(overlap_mat.shape[0]-1)
        elif graph_type=='weighted':
            shuf_structure_index[s_idx] = 1 - np.mean(np.sum(overlap_mat, axis=0))/(overlap_mat.shape[0]-1)

    return structure_index, bin_label, overlap_mat, shuf_structure_index