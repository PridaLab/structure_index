import warnings, copy #,sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#from scipy import sparse, linalg
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
from tqdm.auto import tqdm
import networkx as nx
# overlap_options = ['one_third','continuity']
# graph_options = ['binary', 'weighted']
distance_options = ['euclidean','geodesic']
continuity_kernel_options = ['gaussian']

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
                    raise TypeError(f"{fname}(...): arg '{argname}': type is"+\
                                    f" {type(argval)}, must be {argtype}")
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


def create_ndim_grid(label, n_bins, min_label, max_label, discrete_label):
    
    ndims = label.shape[1]
    grid_edges = list()
    for nd in range(ndims):
        if discrete_label[nd]:
            grid_edges.append(np.tile(np.unique(label[:,nd]).reshape(-1,1),(1,2)))
        else:
            steps = (max_label[nd] - min_label[nd])/n_bins[nd]
            edges = np.linspace(min_label[nd], max_label[nd], 
                                                    n_bins[nd]+1).reshape(-1,1)
            grid_edges.append(np.concatenate((edges[:-1], edges[1:]), axis = 1))

    #Generate the grid containing the indices of the points of the label and 
    #the coordinates as the mid point of edges
    grid = np.empty([e.shape[0] for e in grid_edges], object)
    mesh = meshgrid2(tuple([np.arange(s) for s in grid.shape]))
    meshIdx = np.vstack([col.ravel() for col in mesh]).T
    coords = np.zeros(meshIdx.shape+(3,))
    grid = grid.ravel()

    for elem, idx in enumerate(meshIdx):
        logic = np.zeros(label.shape[0])
        for dim in range(len(idx)):
            min_edge = grid_edges[dim][idx[dim],0]
            max_edge = grid_edges[dim][idx[dim],1]
            logic = logic + 1*np.logical_and(label[:,dim]>=min_edge,label[:,dim]<=max_edge)
            coords[elem,dim,0] = min_edge
            coords[elem,dim,1] = 0.5*(min_edge + max_edge)
            coords[elem,dim,2] = max_edge
        grid[elem] = list(np.where(logic == meshIdx.shape[1])[0])
        
    return grid, coords


def cloud_overlap_radius(cloud1, cloud2, r, distance_metric):
    """Compute overlapping between two clouds of points.
    
    Parameters:
    ----------
        cloud1: numpy 2d array of shape [n_samples_1,n_features]
            Array containing the cloud of points 1

        cloud2: numpy 2d array of shape [n_samples_2,n_features]
            Array containing the cloud of points 2

        k: int
            Number of neighbors used to compute the overlapping between 
            bin-groups. This parameter controls the tradeoff between local 
            and global structure.

        distance_metric: str
            Type of distance used to compute the closest n_neighbors. See 
            'distance_options' for currently supported distances.

        overlap_method: str (default: 'one_third')
            Type of method use to compute the overlapping between bin-groups. 
            See 'overlap_options' for currently supported methods.

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
    cloud_label = np.hstack((np.ones(cloud1.shape[0]),np.ones(cloud2.shape[0])*2))

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
        #total fraction of neighbors that belong to the other cloud
    overlap_1_2 = np.sum(I[:idx_sep,:]>=idx_sep)/np.sum(num_neigh[:idx_sep])
    overlap_2_1 = np.sum(I[idx_sep:,:]<idx_sep)/np.sum(num_neigh[idx_sep:])

    return overlap_1_2, overlap_2_1


def cloud_overlap_neighbors(cloud1, cloud2, k, distance_metric):
    """Compute overlapping between two clouds of points.
    
    Parameters:
    ----------
        cloud1: numpy 2d array of shape [n_samples_1,n_features]
            Array containing the cloud of points 1

        cloud2: numpy 2d array of shape [n_samples_2,n_features]
            Array containing the cloud of points 2

        k: int
            Number of neighbors used to compute the overlapping between 
            bin-groups. This parameter controls the tradeoff between local 
            and global structure.

        distance_metric: str
            Type of distance used to compute the closest n_neighbors. See 
            'distance_options' for currently supported distances.

        overlap_method: str (default: 'one_third')
            Type of method use to compute the overlapping between bin-groups. 
            See 'overlap_options' for currently supported methods.

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
        #total fraction of neighbors that belong to the other cloud
    overlap_1_2 = np.sum(I[:idx_sep,:]>=idx_sep)/(cloud1.shape[0]*k)
    overlap_2_1 = np.sum(I[idx_sep:,:]<idx_sep)/(cloud2.shape[0]*k)

    return overlap_1_2, overlap_2_1


@validate_args_types(data=np.ndarray, label=np.ndarray, n_bins=(int,np.integer,list), 
    dims=(type(None),list), distance_metric=str, n_neighbors=(int,np.integer),
    num_shuffles=(int,np.integer), discrete_label=(list,bool), continuity_kernel=(type(None),str,np.ndarray), 
    verbose=bool)

def compute_structure_index(data, label, n_bins=10, dims=None, **kwargs):
    '''compute structure index main function
    
    Parameters:
    ----------
        data: numpy 2d array of shape [n_samples,n_dimensions]
            Array containing the signal

        label: numpy 2d array of shape [n_samples,n_features]
            Array containing the labels of the data. It can either be a 
            column vector (scalar feature) or a 2D array (vectorial feature)
        
    Optional parameters:
    --------------------
        n_bins: integer (default: 10)
            number of bin-groups the label will be divided into (they will 
            become nodes on the graph). For vectorial features, if one wants 
            different number of bins for each entry then specify n_bins as a 
            list (i.e. [10,20,5]). Note that it will be ignored if 
            'discrete_label' is set to True.

        dims: list of integers or None (default: None)
            list of integers containing the dimensions of data along which the 
            structure index will be computed. Provide None to compute it along 
            all dimensions of data.
        
        distance_metric: str (default: 'euclidean')
            Type of distance used to compute the closest n_neighbors. See 
            'distance_options' for currently supported distances.

        n_neighbors: int (default: 15)
            Number of neighbors used to compute the overlapping between 
            bin-groups. This parameter controls the tradeoff between local and 
            global structure.

        discrete_label: boolean (default: False)
            If the label is discrete, then one bin-group will be created for 
            each discrete value it takes. Note that if set to True, 'n_bins' 
            parameter will be ignored.
        
        num_shuffles: int (default: 100)
            Number of shuffles to be computed. Note it must fall within the 
            interval [0, np.inf).
        
        continuity_kernel: None/str/list/np.ndarray (default: None)
            Kernel to apply to the overlapping matrix to evaluate continuity
            of the feature.

        verbose: boolean (default: False)
            Boolean controling whether or not to print internal process.

                           
    Returns:
    -------
        SI: float
            structure index

        bin_label: tuple
            Tuple containing:
                [0] Array indicating the bin-group to which each data point has 
                    been assigned.
                [1] Array indicating feature limits of each bin-group. Size is
                [number_bin_groups, n_features, 3] where the last dimension 
                contains [bin_st, bin_center, bin_en]

        overlap_mat: numpy 2d array of shape [n_bins, n_bins]
            Array containing the overlapping between each pair of bin-groups.

        shuf_SI: numpy 1d array of shape [num_shuffles,]
            Array containing the structure index computed for each shuffling 
            iteration.
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
    #Note input type validity is handled by the decorator. Here the values 
    #themselves are being checked.
    #i) data input
    assert data.ndim==2, "Input 'data' must be a 2D numpy ndarray with shape"+\
        " of samples and m the number of dimensions."
    #ii) label input
    if label.ndim==1:
        label = label.reshape(-1,1)
    assert label.ndim==2,\
        "label must be a 1D or 2D array."
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
            f"'distance_metric'. Valid options are {distance_options}."
    else:
        distance_metric = 'euclidean'
    #ix) n_neighbors input
    if ('n_neighbors' in kwargs) and ('radius' in kwargs):
        raise ValueError("Both n_neighbors and radius provided. Please only"+\
                                                                " specify one")

    if 'radius' in kwargs:
        neighborhood_size = kwargs['radius']
        assert neighborhood_size>0, "Input 'radius' must be larger than 0"
        cloud_overlap = cloud_overlap_radius
    else:
        if 'n_neighbors' in kwargs:
            neighborhood_size = kwargs['n_neighbors']
            assert neighborhood_size>2, "Input 'n_neighbors' must be larger"+\
                                                                    "than 2."
        else:
            neighborhood_size = 15
        cloud_overlap = cloud_overlap_neighbors

    #x) discrete_label input
    if 'discrete_label' in kwargs:
        discrete_label = kwargs['discrete_label']
        if isinstance(discrete_label,bool):
            discrete_label = [discrete_label for idx in range(label.shape[1])]
        else:
            assert np.all([isinstance(idx, bool) for idx in discrete_label]),\
            "Input 'discrete_label' must be boolean or list of booleans."
    else:
        discrete_label = [False for idx in range(label.shape[1])]

    #xi) num_shuffles input
    if 'num_shuffles' in kwargs:
        num_shuffles = kwargs['num_shuffles']
        assert num_shuffles>=0, "Input 'num_shuffles must fall within the "+\
                                                        "interval [0, np.inf)"
    else:
        num_shuffles = 100
    #xii) verbose input
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = False
    #xiii) continuit_kernel
    if 'continuity_kernel' in kwargs:
        continuity_kernel = kwargs['continuity_kernel']
        #continuity kernel for vectorial features not implemented
        if continuity_kernel and label.shape[1]>1:
            warnings.warn(f"'continuity_kernel' is not currently supported for "
                        f"vectorial features. Disabling it (continuity agnostic).")
            continuity_kernel = None

        if isinstance(continuity_kernel, str):
            assert continuity_kernel in continuity_kernel_options, f"Invalid input "+\
                f"'continuity_kernel'. Valid options are {continuity_kernel_options}."
            #continuity kernel for discrete labels not implemented.
            if continuity_kernel and discrete_label:
                warnings.warn(f"'continuity_kernel' for discrete labels assumes ordered unique labels.")

            if continuity_kernel=='gaussian':
                #create gaussian kernel
                gaus = lambda x,x0,sig: np.exp(-(((x-x0)**2)/(2*sig**2)));
                if 'gaussian_sigma' in kwargs:
                    sigma = kwargs['gaussian_sigma']
                else:
                    sigma = 0.5*n_bins[0] / (2 * np.sqrt(2 * np.log(2)))
                if 'continuity_lambda' in kwargs:
                    continuity_lambda = kwargs['continuity_lambda']
                else: continuity_lambda = 0.1;
                kernel_gauss = np.zeros((n_bins[0], n_bins[0]))*np.nan
                for node in range(n_bins[0]):
                    node_gauss = 1 - gaus(np.arange(n_bins[0]),node,sigma)
                    kernel_gauss[node,:] = (n_bins[0]-1)*node_gauss/(continuity_lambda*np.nansum(node_gauss))
                continuity_kernel = kernel_gauss

        elif isinstance(continuity_kernel, np.ndarray):
            assert continuity_kernel.ndim==2, "'continuity_kernel', must be a square matrix with "+\
                                                        "shape equal to the number of bins"
            assert continuity_kernel.shape[0]==continuity_kernel.shape[1], "'continuity_kernel', must be a square matrix with "+\
                                                        "shape equal to the number of bins"
            assert continuity_kernel.shape[0]==n_bins[0], "'continuity_kernel' shape does not match "+\
                                                        "number of bins"
    else:
        continuity_kernel = None

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
    if verbose: print('Computing bin-groups...', sep='', end = '')
    #a) Check bin-num vs num unique label
    for dim in range(label.shape[1]):
        num_unique_label =len(np.unique(label[:,dim]))
        if discrete_label[dim]:
            n_bins[dim] = num_unique_label
        elif n_bins[dim]>num_unique_label:
             warnings.warn(f"Along column {dim}, input 'label' has less or the same unique "
                            f"values ({num_unique_label}) than specified in "
                            f"'n_bins' ({n_bins[dim]}). Changing 'n_bins' to "
                            f"{num_unique_label} and setting it to discrete.")
             n_bins[dim] = num_unique_label
             discrete_label[dim] = True

    #b) Create bin edges of bin-groups
    if 'min_label' in kwargs:
        min_label = kwargs['min_label']
        if not isinstance(min_label, list):
            min_label = [min_label for nb in range(label.shape[1])]
    else:
        min_label = np.percentile(label,5, axis = 0)
        if any(discrete_label): 
            min_label[discrete_label] = np.min(label[:,discrete_label]) 
    if 'max_label' in kwargs:
        max_label = kwargs['max_label']
        if not isinstance(max_label, list):
            max_label = [max_label for nb in range(label.shape[1])]
    else:
        max_label = np.percentile(label,95, axis = 0)
        if any(discrete_label):
            max_label[discrete_label] = np.max(label[:,discrete_label]) 

    for ld in range(label.shape[1]): #prevent rounding problems
        label[np.where(label[:,ld]<min_label[ld])[0],ld] = min_label[ld]+0.00001
        label[np.where(label[:,ld]>max_label[ld])[0],ld] = max_label[ld]-0.00001

    grid, coords = create_ndim_grid(label, n_bins, min_label, max_label, discrete_label)
    bin_label = np.zeros(label.shape[0],).astype(int)*np.nan
    for b in range(len(grid)):
        bin_label[grid[b]] = b

    #iv). Clean outliers from each bin-groups if specified in kwargs
    if 'filter_noise' in kwargs and kwargs['filter_noise']:
        for l in range(len(grid)):
            noise_idx = filter_noisy_outliers(data[bin_label==l,:])
            noise_idx = np.where(bin_label==l)[0][noise_idx]
            bin_label[noise_idx] = 0

    #v). Discard outlier bin-groups (n_points < n_neighbors)
    #a) Compute number of points in each bin-group
    unique_bin_label = np.unique(bin_label[~np.isnan(bin_label)])
    n_points = np.array([np.sum(bin_label==val) for val in unique_bin_label])

    #b) Get the bin-groups that do not meet criteria and delete them
    min_points_per_bin = 0.1*data.shape[0]/np.prod(n_bins)
    del_labels = np.where(n_points<min_points_per_bin)[0]

    #c) delete outlier bin-groups
    for del_idx in del_labels:
        bin_label[bin_label==unique_bin_label[del_idx]] = np.nan
    #d) re-computed valid bins
    unique_bin_label = np.unique(bin_label[~np.isnan(bin_label)])
    if verbose:
            print('\b\b\b: Done')

    #e) delete bins from continuity_kernel if applicable
    if not isinstance(continuity_kernel, type(None)):
        continuity_kernel = np.delete(continuity_kernel, del_labels, 0)
        continuity_kernel = np.delete(continuity_kernel, del_labels, 1)

    #__________________________________________________________________________
    #|                                                                        |#
    #|                       2. COMPUTE STRUCTURE INDEX                       |#
    #|________________________________________________________________________|#
    #i). compute overlap between bin-groups pairwise
    num_bins = len(unique_bin_label)
    overlap_mat = np.zeros((num_bins, num_bins))*np.nan
    if verbose: 
        bar=tqdm(total=int((num_bins**2-num_bins)/2), desc='Computing overlap')
    for a in range(num_bins):
        A = data[bin_label==unique_bin_label[a]]
        for b in range(a+1, num_bins):
            B = data[bin_label==unique_bin_label[b]]
            overlap_a_b, overlap_b_a = cloud_overlap(A,B,neighborhood_size, 
                                                                distance_metric)
            overlap_mat[a,b] = overlap_a_b
            overlap_mat[b,a] = overlap_b_a
            if verbose: bar.update(1)  
    if verbose: bar.close()
    #i.bis) Apply continuity kernel if applicable
    if not isinstance(continuity_kernel, type(None)):
        overlap_mat = continuity_kernel*overlap_mat
    #ii). compute structure_index (SI)
    if verbose: print('Computing structure index...', sep='', end = '')
    degree_nodes = np.nansum(overlap_mat, axis=1)
    SI = 1 - 2*np.nansum(degree_nodes)/(num_bins*(num_bins-1))
    SI = np.max([SI, 0])
    if verbose: print(f"\b\b\b: {SI:.2f}")
    #iii). Shuffling
    shuf_SI = np.zeros((num_shuffles,))*np.nan
    shuf_overlap_mat = np.zeros((overlap_mat.shape))
    if verbose: bar=tqdm(total=num_shuffles,desc='Computing shuffling')
    for s_idx in range(num_shuffles):
        shuf_bin_label = copy.deepcopy(bin_label)
        np.random.shuffle(shuf_bin_label)
        shuf_overlap_mat *= np.nan
        for a in range(shuf_overlap_mat.shape[0]):
            A = data[shuf_bin_label==unique_bin_label[a]]
            for b in range(a+1, shuf_overlap_mat.shape[1]):
                B = data[shuf_bin_label==unique_bin_label[b]]
                overlap_a_b, overlap_b_a = cloud_overlap(A,B, 
                                            neighborhood_size, distance_metric)
                shuf_overlap_mat[a,b] = overlap_a_b
                shuf_overlap_mat[b,a] = overlap_b_a
        #iii) apply continuity kernel
        if not isinstance(continuity_kernel, type(None)):
            shuf_overlap_mat = continuity_kernel*shuf_overlap_mat
        #iii) compute structure_index (SI)
        degree_nodes = np.nansum(shuf_overlap_mat, axis=1)
        shuf_SI[s_idx] = 1 - 2*np.nansum(degree_nodes)/(num_bins*(num_bins-1))
        shuf_SI[s_idx] = np.max([shuf_SI[s_idx], 0])
        if verbose: bar.update(1) 
    if verbose: bar.close()
    if verbose and num_shuffles>0:
        print(f"Shuffling 99th percentile: {np.percentile(shuf_SI,99):.2f}")

    return SI, (bin_label,coords), overlap_mat, shuf_SI


def draw_graph(overlap_mat, ax, node_cmap = plt.cm.tab10, edge_cmap = plt.cm.Greys, **kwargs):
    """Draw weighted directed graph from overlap matrix.
    
    Parameters:
    ----------
        overlap_mat: numpy 2d array of shape [n_bins, n_bins]
            Array containing the overlapping between each pair of bin-groups.

        ax: matplotlib pyplot axis object.

    Optional parameters:
    --------------------
        node_cmap: pyplot colormap (default: plt.cm.tab10)
            colormap for mapping nodes.

        edge_cmap: pyplot colormap (default: plt.cm.Greys)
            colormap for mapping intensities of edges.

        node_cmap: pyplot colormap (default: plt.cm.tab10)
            pyplot colormap used to color the nodes of the graph.

        node_size: scalar or array  (default: 1000)
            size of nodes. If an array is specified it must be the same length 
            as nodelist.

        scale_edges: scalar (default: 5)
            number used to scale the width of the edges.

        edge_vmin: scalar (default: 0)
            minimum  for edge colormap scaling

        edge_vmax: scalar (default: 0.5)
            maximum for edge colormap scaling

        node_names: scalar (default: 0)
            list containing name of nodes. If numerical, then nodes colormap 
            will be scale according to it.

        node_color: list of colors (default: False)
            A list of node colors to be used instead of a colormap. 
            It must be the same length as nodelist.
            If not specified it defaults to False (bool) and uses `node_cmap` instead

    """
    if int(nx.__version__[0])<3:
        g = nx.from_numpy_matrix(overlap_mat,create_using=nx.DiGraph)
    else:
        g = nx.from_numpy_array(overlap_mat,create_using=nx.DiGraph) #version update function


    number_nodes = g.number_of_nodes()
    
    if 'node_size' in kwargs:
        node_size = kwargs['node_size']
    else:
        node_size = 800

    if 'scale_edges' in kwargs:
        scale_edges = kwargs['scale_edges']
    else:
        scale_edges = 5

    if 'edge_vmin' in kwargs:
        edge_vmin = kwargs['edge_vmin']
    else:
        edge_vmin = 0

    if 'edge_vmax' in kwargs:
        edge_vmax = kwargs['edge_vmax']
    else:
        edge_vmax = 0.5

    if 'node_color' in kwargs:
        node_color = kwargs['node_color']
    else:
        node_color = False

    if 'arrow_size' in kwargs:
        arrow_size = kwargs['arrow_size']
    else:
        arrow_size = 20

    if 'node_names' in kwargs:
        node_names = kwargs['node_names']
        nodes_info = list(g.nodes(data=True))
        names_dict = {val[0]: node_names[i] for i, val in enumerate(nodes_info)}
        with_labels = True
        if not isinstance(node_names[0], str):
            node_val = node_names
        else:
            node_val = range(number_nodes)
    else:
        names_dict = dict()
        node_val = range(number_nodes)
        with_labels = False

    if 'layout_type' in kwargs:
        layout_type = kwargs['layout_type']
    else:
        layout_type = nx.circular_layout
        
    if not node_color: # obtain list of colors from cmap
        norm_cmap = matplotlib.colors.Normalize(vmin=np.min(node_val), vmax=np.max(node_val))
        node_color = list()
        for ii in range(number_nodes):
          #colormap possible values = viridis, jet, spectral
          node_color.append(np.array(node_cmap(norm_cmap(node_val[ii]),bytes=True))/255)

    widths = nx.get_edge_attributes(g, 'weight')

    wdg = nx.draw_networkx(g, pos=layout_type(g), node_size=node_size, 
            node_color=node_color, width=np.array(list(widths.values()))*scale_edges, 
            edge_color= np.array(list(widths.values())), edge_cmap =edge_cmap, 
            arrowsize = arrow_size, edge_vmin = edge_vmin, edge_vmax = edge_vmax, labels = names_dict,
            arrows=True ,connectionstyle="arc3,rad=0.15", with_labels = with_labels, ax=ax)
    
    return wdg