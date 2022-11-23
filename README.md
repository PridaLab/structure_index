# structure_index

Identifying the structured distribution (or lack thereof) of a given feature over a point cloud is a general research question. In the neuroscience field, this problem arises while investigating representations over neural manifolds (e.g., spatial coding), in the analysis of neurophysiological signals (e.g., auditory coding) or in anatomical image segmentation. 

The Structure Index is a graph-based topological metric aimed to quantify the distribution of feature values in arbitrary dimensional spaces. The SI is defined from the overlapping distribution of data points sharing similar feature values in a given neighborhood. It can be applied to both scalar and vectorial features permitting quantification of the relative contribution of related variables.

![sI_github_F1](https://user-images.githubusercontent.com/48024498/203568627-fd912bb2-fc94-4c1f-bfe3-85247dc1cde5.png)

Illustration of the concepts behind the definition of the Structure Index (SI). A, Feature gradient distribution in a 2D-ellipsoid data cloud. Each point in the data cloud is assigned to a group associated with a feature bin value (bin-group). B, C, Next, the overlapping matrix between bin-groups is computed. D, The overlapping matrix represents a connection graph between bin-groups, where structure (overlapping, clustering, etc..) can be quantified using the SI from 0 (random, equivalent to full overlapping) to 1 (maximal separation, equivalent to zero overlapping between bins). E, The case of a random distribution in a 2D data cloud.
