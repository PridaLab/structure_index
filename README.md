# Structure Index

Welcome! This repository hosts the implementation for the Structure Index (SI), a graph-based topological metric able to quantify the amount of structure present at the distribution of a given feature over a point cloud in an arbitrary D-dimensional space.

## The method

Identifying the structure (or lack thereof) of the distribution of a given feature over a point cloud is a general research question. In the neuroscience field, this problem arises while investigating representations over neural manifolds (e.g., spatial coding), in the analysis of neurophysiological signals (e.g., auditory coding) or in anatomical image segmentation. 

The SI is defined from the overlapping distribution of data points sharing similar feature values in a given neighborhood. It can be applied to both scalar and vectorial features permitting quantification of the relative contribution of related variables. The following image illustrates the concepts behind this method:

![sI_github_F1](https://user-images.githubusercontent.com/48024498/203568627-fd912bb2-fc94-4c1f-bfe3-85247dc1cde5.png)

**A**, Feature gradient distribution in a 2D-ellipsoid data cloud. Each point in the data cloud is assigned to a group associated with a feature bin value (bin-group). **B**, **C**, Next, the overlapping matrix between bin-groups is computed. **D**, The overlapping matrix represents a connection graph between bin-groups, where structure (overlapping, clustering, etc..) can be quantified using the SI from 0 (random, equivalent to full overlapping) to 1 (maximal separation, equivalent to zero overlapping between bins). **E**, The case of a randomly distributed feature in a 2D data cloud.
 
## How to use it
```
structure_index, bin_label, overlap_mat, shuf_structure_index = compute_structure_index(data, label)
```

### Parameters
        data: numpy 2d array of shape [n_samples,n_dimensions]
            Array containing the signal

        label: numpy 2d array of shape [n_samples,n_features]
            Array containing the labels of the data. It can either be a column vector (scalar feature) 
            or a 2D array (vectorial feature)

### Optional parameters
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
            If the label is discrete one bin-group will be created for each discrete value it 
            takes. Note that if set to True, 'n_bins' parameter will be ignored.
        
        num_shuffles: int (default: 100)
            Number of shuffles to be computed. Note it must fall within the interval [0, np.inf).

        verbose: boolean (default: False)
            Boolean controling whether or not to print internal process.
            
### Returns:
        structure_index: float
            Computed structure index

        bin_label: numpy 1d array of shape [n_samples,]
            Array containing the bin-group to which each data point has been assigned.

        overlap_mat: numpy 2d array of shape [n_bins, n_bins]
            Array containing the overlapping between each pair of bin-groups.

        shuf_structure_index: numpy 1d array of shape [num_shuffles,]
            Array containing the structure index computed for each shuffling iteration.

