import sys, warnings, copy
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, linalg
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    use_fast = True
except:
    use_fast = False

from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from decorator import decorator

overlap_options = ['one_third','continuity']
graph_options = ['binary', 'weighted']
distance_options = ['euclidean','geodesic']

def validate_args_types(**decls):
    """Decorator to check argument types.

    Usage:

    @check_args(name=str, text=(int,str))
    def parse_rule(name, text): ...
    """
    @decorator
    def wrapper(func, *args, **kwargs):
        code = func.__code__
        fname = func.__name__
        names = code.co_varnames[:code.co_argcount]
        for argname, argtype in decls.items():
            arg_provided = True
            if argname in names:
                argval = args[names.index(argname)]
            elif argname in kwargs:
                argval = kwargs.get(argname)
            else:
                arg_provided = False
            if arg_provided:
                if not isinstance(argval, argtype):
                    raise TypeError(f"{fname}(...): arg '{argname}': type is "+\
                                    f"{type(argval)}, must be {argtype}")
        return func(*args, **kwargs)
    return wrapper


def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,1), axis=1) - 1
    noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    return noiseIdx


def compute_cloud_overlap(cloud1, cloud2, k, distance_metric, overlap_method):
    """Compute overlapping between two clouds of points.
    
    Parameters:
    ----------
        cloud1: numpy 2d array of shape [n_samples_1,n_features]
            Array containing the cloud of points 1

        cloud2: numpy 2d array of shape [n_samples_2,n_features]
            Array containing the cloud of points 2

        k: int
            Number of neighbors used to compute the overlapping between bin-groups. This parameter 
            controls the tradeoff between local and global structure.

        distance_metric: str
            Type of distance used to compute the closest n_neighbors. See 'distance_options' for 
            currently supported distances.

        overlap_method: str (default: 'one_third')
            Type of method use to compute the overlapping between bin-groups. See 'overlap_options'
            for currently supported methods.

    Returns:
    -------
        overlap_1_2: float
            Degree of overlapping of cloud1 over cloud2

        overlap_1_2: float
            Degree of overlapping of cloud2 over cloud1         

    """
    #Stack both clouds
    cloud_all = np.vstack((cloud1, cloud2)).astype('float32')
    idx_sep = cloud1.shape[0]
    #Create cloud label
    cloud_label = np.hstack((np.ones(cloud1.shape[0]), np.ones(cloud2.shape[0])*2))

    #Compute k neighbours graph
    if distance_metric == 'euclidean':
        if use_fast:
            index = faiss.IndexFlatL2(cloud_all.shape[1])   # build the index
            index.add(cloud_all) # add vectors to the index
            _, I = index.search(cloud_all, k+1) # sanity check
            I = I[:,1:]
        else:
            knn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(cloud_all)
            I = knn.kneighbors(return_distance=False)

    elif distance_metric == 'geodesic':
        model_iso = Isomap(n_components = 1)
        emb = model_iso.fit_transform(cloud_all)
        dist_mat = model_iso.dist_matrix_
        knn = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(dist_mat)
        I = knn.kneighbors(return_distance=False)

    #Compute overlapping
    if overlap_method == 'continuity': #total fraction of neighbors that belong to the other cloud
        overlap_1_2 = np.sum(I[:idx_sep,:]>=idx_sep)/(cloud1.shape[0]*k)
        overlap_2_1 = np.sum(I[idx_sep:,:]<idx_sep)/(cloud2.shape[0]*k)
    elif overlap_method == 'one_third':
        #Compute overlap threshold for each individual point
        overlap_th = k/3
        degree_1 = np.sum(I[:idx_sep,:]>=idx_sep, axis=1)
        overlap_1_2 = np.sum(degree_1 >= overlap_th)/degree_1.shape[0]
        degree_2 = np.sum(I[idx_sep:,:]<idx_sep, axis=1)
        overlap_2_1 = np.sum(degree_2 >= overlap_th)/degree_2.shape[0]

    return overlap_1_2, overlap_2_1


@validate_args_types(data=np.ndarray, label=np.ndarray, nbins=(int,np.integer), 
    dims=(type(None),list), distance_metric=str, n_neighbors=(int,np.integer),
    num_shuffles=(int,np.integer), discrete_bin_label=bool, verbose=bool)
def compute_structure_index(data, label, n_bins=10, dims=None, **kwargs):
    '''compute structure index main function
    
    Parameters:
    ----------
        data: numpy 2d array of shape [n_samples,n_features]
            Array containing the signal

        label: numpy 1d array of shape [n_samples,]
            Array containing the labels of the data.
        
    Optional parameters:
    --------------------
        n_bins: integer (default: 10)
            number of bin-groups the label will be divided into (they will become nodes on the 
            graph). Note that it will be ignored if 'discrete_bin_label' is set to True.

        dims: list of integers or None (default: None)
            list of integers containing the dimensions of data along which the structure index will
            be computed. Provide None to compute it along all dimensions of data.
        
        distance_metric: str (default: 'euclidean')
            Type of distance used to compute the closest n_neighbors. See 'distance_options' for 
            currently supported distances.

        overlap_method: str (default: 'one_third')
            Type of method use to compute the overlapping between bin-groups. See 'overlap_options'
            for currently supported methods.

        graph_type: str (default: 'weighted')
            Type of graph used to compute structure index. Either 'weighted' or 'binary'. 

        n_neighbors: int (default: 3)
            Number of neighbors used to compute the overlapping between bin-groups. This parameter 
            controls the tradeoff between local and global structure.

        discrete_bin_label: boolean (default: False)
            If the label is discrete, then one bin-group will be created for each discrete value it 
            takes. Note that if set to True, 'n_bins' parameter will be ignored.
        
        num_shuffles: int (default: 100)
            Number of shuffles to be computed. Note it must fall within the interval [0, np.inf).

        verbose: boolean (default: False)
            Boolean controling whether or not to print internal process.

                           
    Returns:
    -------
        structure_index: float
            Computed structure index

        bin_label: numpy 1d array of shape [n_samples,]
            Array containing the bin-group to which each data point has been assigned.

        overlap_mat: numpy 2d array of shape [n_bins, n_bins]
            Array containing the overlapping between each pair of bin-groups.

        shuf_structure_index: numpy 1d array of shape [num_shuffles,]
            Array containing the structure index computed for each shuffling iteration.
    '''

    #TODO:
        #distance_metric cosyne?
        #re-evaluate which arguments put outside whichones in kwargs
        #write function definition
        #check all imports are being used
        #include plot-function
        #check n-neighbours vs num points per bin
    #__________________________________________________________________________
    #|                                                                        |#
    #|                        0. CHECK INPUT VALIDITY                         |#
    #|________________________________________________________________________|#
    #Note input type validity is handled by the decorator. Here it the values 
    #themselves are being checked.
    #i) data input
    assert data.ndim==2, "Input 'data' must be a 2D numpy ndarray with shape "+\
        " of samples and m the number of dimensions."

    #ii) label input
    if label.ndim==2 and label.shape[1] == 1: #if 2D transform into column vector
            label = label[:,0]
    assert label.ndim==1,\
        "label must be a 1D array (or 2D with only one column)."
    #iii) n_bins input
    assert n_bins>1, "Input 'n_bins' must be an integer larger than 1."
    #iv) dims input
    if isinstance(dims, type(None)): #if dims is None, then take all dimensions
        dims = list(range(data.shape[1]))
    #v) distance_metric
    if 'distance_metric' in kwargs:
        distance_metric = kwargs['distance_metric']
        assert distance_metric in distance_options, f"Invalid input "+\
            "'distance_metric'. Valid options are {distance_options}."
    else:
        distance_metric = 'euclidean'
    #vi) overlap_method input
    if 'overlap_method' in kwargs:
        overlap_method = kwargs['overlap_method']
        assert overlap_method in overlap_options, f"Invalid input "+\
            "'overlap_method'. Valid options are {overlap_options}."
    else:
        overlap_method = 'one_third'
    #vii) graph_type input
    if 'graph_type' in kwargs:
        graph_type = kwargs['graph_type']
        assert graph_type in graph_options, f"Invalid input 'graph_type'. "+\
            "Valid options are {graph_options}."
    else:
        graph_type = 'binary'
    #viii) overlap_threshold input
    if graph_type == 'binary':
        if 'overlap_threshold' in kwargs:
            overlap_threshold = kwargs['overlap_threshold']
            assert overlap_threshold>0 and overlap_threshold<=1, \
                "Input 'overlap_threshold' must belong to interval (0,1]."
        else:
            overlap_threshold = 0.5
    elif graph_type == 'weighted' and 'overlap_threshold' in kwargs:
         warnings.warn(f"Input 'graph_type' is not 'binary' ('{graph_type}') "
                "but input 'overlap_threshold' provided. It will be ignored.")
    #ix) n_neighbors input
    if 'n_neighbors' in kwargs:
        n_neighbors = kwargs['n_neighbors']
        assert n_neighbors>2, "Input 'n_neighbors' must be larger than 2."
    else:
        n_neighbors = 3
    #x) discrete_bin_label input
    if 'dicrete_bin_label' in kwargs:
        discrete_bin_label = kwargs['dicrete_bin_label']
    else:
        discrete_bin_label = False
    #xi) num_shuffles input
    if 'num_shuffles' in kwargs:
        num_shuffles = kwargs['num_shuffles']
        assert num_shuffles>=0, "Input 'num_shuffles must belong to interval [0, np.inf)"
    else:
        num_shuffles = 100
    #xii) verbose input
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = False
    #__________________________________________________________________________
    #|                                                                        |#
    #|                           1. PREPROCESS DATA                           |#
    #|________________________________________________________________________|#
    #i).Keep only desired dims
    data = data[:,dims]
    if data.ndim == 1: #if left with 1 dim, keep the 2D shape
        data = data.reshape(-1,1)

    #ii).Delete nan values from label and data
    data_nans = np.any(np.isnan(data), axis = 1)
    label_nans = np.isnan(label)
    delete_nans = np.where(data_nans+label_nans)[0]

    data = np.delete(data,delete_nans, axis=0)
    label = np.delete(label,delete_nans, axis=0)

    #iii).Binarize label
    if verbose:
        print('Computing bin-groups...', sep='', end = '')
    if discrete_bin_label: #if discrete label
        bin_label = np.zeros(label.shape)
        unique_label = np.unique(label)
        for b in unique_label:
            bin_label[label == b] = 1 + int(np.max(bin_label))

    else: #if continuous label
        #a) Check bin-num vs num unique label
        num_unique_label = len(np.unique(label))
        if n_bins>num_unique_label:
             warnings.warn(f"Input 'label' has less unique values ({num_unique_label}) than specified "
                            f"'n_bins' ({n_bins}). Changing 'n_bins' to {num_unique_label}.")
             n_bins = num_unique_label

        #b) Create bin edges of bin-groups
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

        bin_size = (max_label - min_label)/ n_bins
        bin_edges_left = np.linspace(min_label,min_label+bin_size*(n_bins-1),n_bins)
        bin_edges = np.column_stack((bin_edges_left, bin_edges_left+bin_size))

        #c) Create bin_label
        bin_label = np.zeros(label.shape)
        for b in range(n_bins-1):
            points_in_bin = np.logical_and(label>=bin_edges[b,0], label<bin_edges[b,1])
            bin_label[points_in_bin] = 1 + int(np.max(bin_label))
        points_in_last_bin = np.logical_and(label>=bin_edges[n_bins-1,0], label<=bin_edges[n_bins-1,1])
        bin_label[points_in_last_bin] = 1 + int(np.max(bin_label))

    #iv). Clean outliers from each bin-groups if specified in kwargs
    if 'filter_noise' in kwargs and kwargs['filter_noise']:
        for l in np.unique(bin_label):
            noise_idx = filter_noisy_outliers(data[bin_label==l,:])
            noise_idx = np.where(bin_label==l)[0][noise_idx]
            bin_label[noise_idx] = 0

    #v). Discard outlier bin-groups (n_points < n_neighbors)
    #a) Compute number of points in each bin-group
    n_points = np.array([np.sum(bin_label==value) for value in np.unique(bin_label)])
    #b) Get the bin-groups that meet criteria and delete them
    del_labels = np.where(n_points < n_neighbors)[0]
    #c) delete outlier bin-groups
    for del_idx in del_labels:
        bin_label[bin_label==del_idx+1] = 0
    #d) renumber bin labels from 1 to n bin-groups
    unique_bin_label = np.unique(bin_label)
    if 0 in np.unique(bin_label):
        for idx in range(1,len(unique_bin_label)):
            bin_label[bin_label==unique_bin_label[idx]]= idx
    if verbose:
            print('\b\b\b: Done')
    #__________________________________________________________________________
    #|                                                                        |#
    #|                       2. COMPUTE STRUCTURE INDEX                       |#
    #|________________________________________________________________________|#
    #i) compute overlap between bin-groups pairwise
    overlap_mat = np.zeros((np.sum(np.unique(bin_label)>0), np.sum(np.unique(bin_label)>0)))
    for ii in range(overlap_mat.shape[0]):
        if verbose:
            print(f"Computing overlapping: {ii+1}/{overlap_mat.shape[0]}", end = '\r')
            if ii+1<overlap_mat.shape[0]:
                sys.stdout.write('\033[2K\033[1G')      
        for jj in range(ii+1, overlap_mat.shape[1]):
            overlap_1_2, overlap_2_1 = compute_cloud_overlap(data[bin_label==ii+1], data[bin_label==jj+1], 
                                                        n_neighbors, distance_metric,overlap_method)
            overlap_mat[ii,jj] = overlap_1_2
            overlap_mat[jj,ii] = overlap_2_1
    #ii) symmetrize overlap matrix
    overlap_mat = (overlap_mat + overlap_mat.T) / 2
    #iii) compute structure_index
    if verbose:
        print('\nComputing structure index...', sep='', end = '')
    if graph_type=='binary':
        degree_nodes = np.sum(1*(overlap_mat>=overlap_threshold), axis=0)
        structure_index = 1 - np.mean(degree_nodes)/(overlap_mat.shape[0]-1)
    elif graph_type=='weighted':
        degree_nodes = np.sum(overlap_mat, axis=0)
        structure_index = 1 - np.mean(degree_nodes)/(overlap_mat.shape[0]-1)
        if overlap_method=='continuity':
            structure_index = 2*(structure_index-0.5)
    if verbose:
        print(f"\b\b\b: {structure_index:.2f}")
    #9. Shuffling
    shuf_structure_index = np.zeros((num_shuffles,))*np.nan
    shuf_overlap_mat = np.zeros((overlap_mat.shape))
    for s_idx in range(num_shuffles):
        if verbose:
            print(f"Computing shuffling: {s_idx+1}/{num_shuffles}", end = '\r')
            sys.stdout.write('\033[2K\033[1G')   
        shuf_bin_label = copy.deepcopy(bin_label)
        np.random.shuffle(shuf_bin_label)
        shuf_overlap_mat *= 0
        for ii in range(shuf_overlap_mat.shape[0]):
            for jj in range(ii+1, shuf_overlap_mat.shape[1]):
                overlap_1_2, overlap_2_1 = compute_cloud_overlap(data[shuf_bin_label==ii+1], data[shuf_bin_label==jj+1], 
                                                        n_neighbors, distance_metric,overlap_method)
                shuf_overlap_mat[ii,jj] = overlap_1_2
                shuf_overlap_mat[jj,ii] = overlap_2_1
        #ii) symetrize overlap matrix
        shuf_overlap_mat = (shuf_overlap_mat + shuf_overlap_mat.T) / 2
        #iii) computed structure_index
        if graph_type=='binary':
            degree_nodes = np.sum(1*(shuf_overlap_mat>=overlap_threshold), axis=0)
            shuf_structure_index[s_idx] = 1 - np.mean(degree_nodes)/(shuf_overlap_mat.shape[0]-1)
        elif graph_type=='weighted':
            degree_nodes = np.sum(shuf_overlap_mat, axis=0)
            shuf_structure_index[s_idx] = 1 - np.mean(degree_nodes)/(shuf_overlap_mat.shape[0]-1)
            if overlap_method=='continuity':
                shuf_structure_index[s_idx] = 2*(shuf_structure_index[s_idx]-0.5)
    if verbose and num_shuffles>0:
        print(f"Computing shuffling: {np.percentile(shuf_structure_index, 99):.2f}")

    return structure_index, bin_label, overlap_mat, shuf_structure_index