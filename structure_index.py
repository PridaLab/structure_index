import sys, warnings, copy
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, linalg
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix

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


def meshgrid2(arrs):
    #arrs: tuple with np.arange of shape of all dimensions
    lens = list(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz*=s
    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)
    return tuple(ans)


def create_ndim_grid(label, n_bins, min_label, max_label):
    ndims = label.shape[1]
    #Get grid limits
    grid_limits = [(min_label[ii], max_label[ii]) for ii in range(ndims)]

    #Define the edges for each dimension
    steps = [(grid_limits[ii][1] - grid_limits[ii][0])/n_bins[ii] for ii in range(ndims)]

    grid_edges = [np.hstack((np.linspace(lim[0], lim[0]+s*n, n+1)[:-1].reshape(-1,1),
                            np.linspace(lim[0], lim[0]+s*n, n+1)[1:].reshape(-1,1)
                            )) for lim, n, s in zip(grid_limits, n_bins, steps)]
    
    #Generate the grid containing the indices of the points of the label and the coordinates as the mid point of edges
    grid = np.empty([e.shape[0] for e in grid_edges], object)
    mesh = meshgrid2(tuple([np.arange(s) for s in grid.shape]))
    meshIdx = np.vstack([col.ravel() for col in mesh]).T
    coords = np.zeros(meshIdx.shape)
    grid = grid.ravel()
    for elem, idx in enumerate(meshIdx):
        logic = np.zeros(label.shape[0])
        for dim in range(len(idx)):
            logic = logic + 1*np.logical_and(label[:,dim] >= grid_edges[dim][idx[dim],0], label[:,dim] <= grid_edges[dim][idx[dim],1])
            coords[elem,dim] = grid_edges[dim][idx[dim],0] + (grid_edges[dim][idx[dim],1] - grid_edges[dim][idx[dim],0]) / 2
        grid[elem] = list(np.where(logic == meshIdx.shape[1])[0])

    return grid, coords


def create_ndim_grid_discrete(label):
    # bin_label = np.zeros(label.shape)
    # unique_label = np.unique(label)
    # for b in unique_label:
    #     bin_label[label == b] = 1 + int(np.max(bin_label))

    #Get grid unique vals for each dim
    ndims = label.shape[1]
    grid_unique = [np.unique(label[:,ii]) for ii in range(ndims)]

    #Generate the grid containing the indices of the points of the label and the coordinates as the mid point of edges
    grid = np.empty([e.shape[0] for e in grid_unique], object)
    mesh = meshgrid2(tuple([np.arange(s) for s in grid.shape]))
    meshIdx = np.vstack([col.ravel() for col in mesh]).T
    coords = np.zeros(meshIdx.shape)
    grid = grid.ravel()
    for elem, idx in enumerate(meshIdx):
        logic = np.zeros(label.shape[0])
        for dim in range(len(idx)):
            logic = logic + 1*(label[:,dim] == grid_unique[dim][idx[dim]])
            coords[elem,dim] = grid_unique[dim][idx[dim]]
        grid[elem] = list(np.where(logic == meshIdx.shape[1])[0])

    return grid, coords


def compute_cloud_overlap_radius(cloud1, cloud2, r, distance_metric, overlap_method):
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
        D = distance_matrix(cloud_all, cloud_all,p=2)

    elif distance_metric == 'geodesic':
        model_iso = Isomap(n_components = 1)
        emb = model_iso.fit_transform(cloud_all)
        D = model_iso.dist_matrix_

    I = np.argsort(D, axis = 1)
    for row in range(I.shape[0]):
        D[row,:]  = D[row,I[row,:]]
    I = I[:, 1:].astype('float32')
    D = D[:, 1:]
    I[D>r]= np.nan
    num_neigh = I.shape[0] - np.sum(np.isnan(I), axis = 1).astype('float32') - 1
    #Compute overlapping
    if overlap_method == 'continuity': #total fraction of neighbors that belong to the other cloud
        overlap_1_2 = np.sum(I[:idx_sep,:]>=idx_sep)/np.sum(num_neigh[:idx_sep])
        overlap_2_1 = np.sum(I[idx_sep:,:]<idx_sep)/np.sum(num_neigh[idx_sep:])

    elif overlap_method == 'one_third':
        #Compute overlap threshold for each individual point
        overlap_th = 1/3
        num_neigh[num_neigh==0] = np.nan
        degree_1 = np.sum(I[:idx_sep,:]>=idx_sep, axis=1)/num_neigh[:idx_sep]
        overlap_1_2 = np.sum(degree_1 >= overlap_th)/np.sum(~np.isnan(num_neigh[:idx_sep]))
        degree_2 = np.sum(I[idx_sep:,:]<idx_sep, axis=1)/num_neigh[idx_sep:]
        overlap_2_1 = np.sum(degree_2 >= overlap_th)/np.sum(~np.isnan(num_neigh[idx_sep:]))

    return overlap_1_2, overlap_2_1


def compute_cloud_overlap_neighbors(cloud1, cloud2, k, distance_metric, overlap_method):
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
            _, I = index.search(cloud_all, k+1)
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


@validate_args_types(data=np.ndarray, label=np.ndarray, n_bins=(int,np.integer, list), 
    dims=(type(None),list), distance_metric=str, n_neighbors=(int,np.integer),
    num_shuffles=(int,np.integer), discrete_bin_label=bool, verbose=bool)
def compute_structure_index(data, label, n_bins=10, dims=None, **kwargs):
    '''compute structure index main function
    
    Parameters:
    ----------
        data: numpy 2d array of shape [n_samples,n_dimensions]
            Array containing the signal

        label: numpy 1d array of shape [n_samples,n_features]
            Array containing the labels of the data. It can either be a column vector (scalar feature) 
            or a 2D array (vectorial feature)
        
    Optional parameters:
    --------------------
        n_bins: integer (default: 10)
            number of bin-groups the label will be divided into (they will become nodes on the 
            graph). For vectorial features, if one wants different number of bins for each entry
            then specify n_bins as a list (i.e. [10,20,5]). Note that it will be ignored if 
            'discrete_bin_label' is set to True.

        dims: list of integers or None (default: None)
            list of integers containing the dimensions of data along which the structure index will
            be computed. Provide None to compute it along all dimensions of data.
        
        distance_metric: str (default: 'euclidean')
            Type of distance used to compute the closest n_neighbors. See 'distance_options' for 
            currently supported distances.

        overlap_method: str (default: 'continuity')
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
        #distance_metric: cosyne
        #re-evaluate which arguments to put outside which ones in kwargs
        #include plot-function
        #maybe plot neighbors inside radius distribution when radius provided
        #if radius selected, and no point has neighbors prompt error
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
    if label.ndim==1:
        label = label.reshape(-1,1)
    assert label.ndim==2,\
        "label must be a 1D array (or 2D)."
    #iii) n_bins input
    if isinstance(n_bins, int) or isinstance(n_bins, np.integer):
        assert n_bins>1,\
        "Input 'n_bins' must be an int or list of int larger than 1."
        n_bins = [n_bins for nb in range(label.shape[1])]
    elif isinstance(n_bins, list):
        assert np.all([nb>1 for nb in n_bins]),\
        "Input 'n_bins' must be an int or list of int larger than 1."

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
        overlap_method = 'continuity'
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
                "Input 'overlap_threshold' must fall within the interval (0,1]."
        else:
            overlap_threshold = 0.5
    elif graph_type == 'weighted' and 'overlap_threshold' in kwargs:
         warnings.warn(f"Input 'graph_type' is not 'binary' ('{graph_type}') "
                "but input 'overlap_threshold' provided. It will be ignored.")
    #ix) n_neighbors input
    if ('n_neighbors' in kwargs) and ('radius' in kwargs):
        raise ValueError('Both n_neighbors and radius provided. Please only specify one')

    if 'radius' in kwargs:
        neighborhood_size = kwargs['radius']
        assert neighborhood_size>0, "Input 'radius' must be larger than 0"
        compute_cloud_overlap = compute_cloud_overlap_radius
        min_points_per_bin = 0.1*data.shape[0]/np.prod(n_bins)
    else:
        if 'n_neighbors' in kwargs:
            neighborhood_size = kwargs['n_neighbors']
            assert neighborhood_size>2, "Input 'n_neighbors' must be larger than 2."
        else:
            neighborhood_size = 3
        compute_cloud_overlap = compute_cloud_overlap_neighbors
        min_points_per_bin = neighborhood_size

    #x) discrete_bin_label input
    if 'dicrete_bin_label' in kwargs:
        discrete_bin_label = kwargs['dicrete_bin_label']
    else:
        discrete_bin_label = False
    #xi) num_shuffles input
    if 'num_shuffles' in kwargs:
        num_shuffles = kwargs['num_shuffles']
        assert num_shuffles>=0, "Input 'num_shuffles must fall within the interval [0, np.inf)"
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
    label_nans = np.any(np.isnan(label), axis = 1)
    delete_nans = np.where(data_nans+label_nans)[0]

    data = np.delete(data,delete_nans, axis=0)
    label = np.delete(label,delete_nans, axis=0)

    #iii).Binarize label
    if verbose:
        print('Computing bin-groups...', sep='', end = '')

    if discrete_bin_label: #if discrete label
        grid, coords = create_ndim_grid_discrete(label)
        bin_label = np.zeros(label.shape[0],).astype(int)
        for b in range(len(grid)):
            bin_label[grid[b]] = b+1

    else: #if continuous label
        #a) Check bin-num vs num unique label
        for dim in range(label.shape[1]):
            num_unique_label =len(np.unique(label[:,dim]))
            if n_bins[dim]>num_unique_label:
                 warnings.warn(f"Input 'label' has less unique values ({num_unique_label}) along "
                                f"dim {dim} than specified in 'n_bins' ({n_bins[dim]}). Changing "
                                f" 'n_bins' to {num_unique_label}.")
                 n_bins[dim] = num_unique_label

        #b) Create bin edges of bin-groups
        if 'min_label' in kwargs:
            min_label = kwargs['min_label']
            if not isinstance(min_label, list):
                min_label = [min_label for nb in range(label.shape[1])]
        else:
            min_label = np.percentile(label,5, axis = 0)
        if 'max_label' in kwargs:
            max_label = kwargs['max_label']
            if not isinstance(max_label, list):
                max_label = [max_label for nb in range(label.shape[1])]
        else:
            max_label = np.percentile(label,95, axis = 0)

        for ld in range(label.shape[1]):
            label[np.where(label[:,ld]<min_label[ld])[0],ld] = min_label[ld] + 0.00001
            label[np.where(label[:,ld]>max_label[ld])[0],ld] = max_label[ld] - 0.00001

        grid, coords = create_ndim_grid(label, n_bins, min_label, max_label)
        bin_label = np.zeros(label.shape[0],).astype(int)
        for b in range(len(grid)):
            bin_label[grid[b]] = b+1

    #iv). Clean outliers from each bin-groups if specified in kwargs
    if 'filter_noise' in kwargs and kwargs['filter_noise']:
        for l in np.unique(bin_label):
            noise_idx = filter_noisy_outliers(data[bin_label==l,:])
            noise_idx = np.where(bin_label==l)[0][noise_idx]
            bin_label[noise_idx] = 0

    #v). Discard outlier bin-groups (n_points < n_neighbors)
    #a) Compute number of points in each bin-group
    unique_bin_label = np.unique(bin_label)
    n_points = np.array([np.sum(bin_label==value) for value in unique_bin_label])

    #b) Get the bin-groups that meet criteria and delete them
    del_labels = np.where(n_points<min_points_per_bin)[0]

    #c) delete outlier bin-groups
    for del_idx in del_labels:
        bin_label[bin_label==unique_bin_label[del_idx]] = 0
    #d) renumber bin labels from 1 to n bin-groups
    unique_bin_label = np.unique(bin_label)
    if 0 in np.unique(bin_label):
        for idx in range(1,len(unique_bin_label)):
            bin_label[bin_label==unique_bin_label[idx]]= idx
    else:
        for idx in range(0,len(unique_bin_label)):
            bin_label[bin_label==unique_bin_label[idx]]= idx+1
    if verbose:
            print('\b\b\b: Done')
    #__________________________________________________________________________
    #|                                                                        |#
    #|                       2. COMPUTE STRUCTURE INDEX                       |#
    #|________________________________________________________________________|#
    #i). compute overlap between bin-groups pairwise
    unique_bin_label = np.unique(bin_label)
    unique_bin_label = unique_bin_label[unique_bin_label>0]
    overlap_mat = np.zeros((len(unique_bin_label), len(unique_bin_label)))
    for ii in range(overlap_mat.shape[0]):
        if verbose:
            print(f"Computing overlapping: {ii+1}/{overlap_mat.shape[0]}", end = '\r')
            if ii+1<overlap_mat.shape[0]:
                sys.stdout.write('\033[2K\033[1G')      
        for jj in range(ii+1, overlap_mat.shape[1]):
            overlap_1_2, overlap_2_1 = compute_cloud_overlap(data[bin_label==unique_bin_label[ii]], 
                                                        data[bin_label==unique_bin_label[jj]], 
                                                        neighborhood_size, distance_metric,overlap_method)
            overlap_mat[ii,jj] = overlap_1_2
            overlap_mat[jj,ii] = overlap_2_1
    #ii). compute structure_index
    if verbose:
        print('Computing structure index...', sep='', end = '')
    if graph_type=='binary':
        overlap_mat = (overlap_mat + overlap_mat.T)/2
        degree_nodes = np.sum(1*(overlap_mat>=overlap_threshold), axis=0)
        structure_index = 1 - np.mean(degree_nodes)/(overlap_mat.shape[0]-1)
    elif graph_type=='weighted':
        degree_nodes = np.sum(overlap_mat, axis=0)
        structure_index = 1 - np.mean(degree_nodes)/(overlap_mat.shape[0]-1)
        if overlap_method=='continuity':
            structure_index = 2*(structure_index-0.5)
            structure_index = np.max([structure_index, 0])
    if verbose:
        print(f"\b\b\b: {structure_index:.2f}")
    #iii). Shuffling
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
                                                        neighborhood_size, distance_metric,overlap_method)
                shuf_overlap_mat[ii,jj] = overlap_1_2
                shuf_overlap_mat[jj,ii] = overlap_2_1
        #iii) computed structure_index
        if graph_type=='binary':
            shuf_overlap_mat = (shuf_overlap_mat + shuf_overlap_mat.T)/2
            degree_nodes = np.sum(1*(shuf_overlap_mat>=overlap_threshold), axis=0)
            shuf_structure_index[s_idx] = 1 - np.mean(degree_nodes)/(shuf_overlap_mat.shape[0]-1)
        elif graph_type=='weighted':
            degree_nodes = np.sum(shuf_overlap_mat, axis=0)
            shuf_structure_index[s_idx] = 1 - np.mean(degree_nodes)/(shuf_overlap_mat.shape[0]-1)
            if overlap_method=='continuity':
                shuf_structure_index[s_idx] = 2*(shuf_structure_index[s_idx]-0.5)
                shuf_structure_index[s_idx] = np.max([shuf_structure_index[s_idx], 0])
    if verbose and num_shuffles>0:
        print(f"Computing shuffling: {s_idx+1}/{num_shuffles} - {np.percentile(shuf_structure_index, 99):.2f}")

    return structure_index, bin_label, overlap_mat, shuf_structure_index