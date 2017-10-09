# distutils: language = c++
# distutils: sources = cpp_sprite_assignment.cpp

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "../include/cpp_sprite_assignment.h":
    void get_rg2s_cpp(float* crds,
                  int n_struct,
                  int n_bead,
                  int n_regions, 
                  int* copies_num,
                  float* rg2s,
                  int* copy_idxs,
                  int* min_struct)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_rgs2(np.ndarray[float, ndim=3, mode="c"] crds not None,
             np.ndarray[int, ndim=1, mode="c"] copies_num not None,
             ):
    """
    Compute radiuses of gyrations (Rg) of N structures, each consisting 
    of M beads. Each bead i=0..M-1 can have K(i) alternative locations 
    x_i(0), x_i(1), .., x_i(K(i) - 1).
    The combination of alternative locations which minimizes the Rg is
    selected and returned for each structure.
    Note that this choice is performed by mere enumeration, so this
    scales exponentially with the number of alternative locations. The 
    complexity is O(K(0) x K(1) x ... x K(M-1)).

    Parameters
    ----------
    crds (np.ndarray) : vector of the coordinates, with shape (N x B x 3),
        where N is the number of structures, and B is the total number of
        alternative coordinates, i.e. B = sum({K(i) : i = 0..M-1}). The 
        alternative coordinates should be placed consecutively.
        Call x(s, i, k) the 3D coordinates of the k-th alternative
        for the i-th bead in the s-th structure.
        For example, if we have 2 beads:
         - i=0 with K(0)=2 alternative locations
         - i=1 with K(1)=3 alternative locations
        crd should have the following structure: 
        [
          [ x(0, 0, 0), x(0, 0, 1), x(0, 1, 0), x(0, 1, 1), x(0, 1, 2) ], 
          [ x(1, 0, 0), x(1, 0, 1), x(1, 1, 0), x(1, 1, 1), x(1, 1, 2) ],
          ...
          [ x(N-1, 0, 0), x(N-1, 0, 1), x(N-1, 1, 0), x(N-1, 1, 1), x(N-1, 1, 2) ]
        ]

    copies_num (np.array) : M integers representing the number of copies 
        for each bead [K(0), K(1), .., K(M-1)]. In the case above [2, 3]

    Returns
    -------
    rg2s (np.array) : the squared radius of gyration for each structure
    best_structure (int) : the index of the structure which minimize
        the Rg, i.e. rg2s[best_structure] = min(rg2s)
    copy_idxs (np.ndarray) : a N x M matrix containing the selection 
        of alternative locations for each structure which minimize Rg:
        [
            W(0),
            W(1),
            ...
            W(N-1),
        ]
        where the vector W(s) = [w(0), w(1), .., w(N-1)] is the best 
        combination of alternative positions, and rg2s[s] corresponds
        to the squared Rg computed on the set of coordinates 
        { x(s, i, W(s)[i]) : i=0..M-1 } 

    """
    cdef int n_struct = crds.shape[0] 
    cdef int n_bead = crds.shape[1]
    cdef int best_structure
    cdef int n_regions = copies_num.shape[0]

    cdef np.ndarray[float, ndim=1] rg2s = np.zeros(n_struct, dtype=np.float32)
    cdef np.ndarray[int, ndim=2, mode="c"] copy_idxs = np.zeros((n_struct, n_regions), dtype = np.int32)

    get_rg2s_cpp(&crds[0,0,0], n_struct, n_bead, n_regions, 
                 &copies_num[0], &rg2s[0], &copy_idxs[0,0], &best_structure)
    
    return rg2s, best_structure, copy_idxs 

def compute_gyration_radius(crd, cluster, index, copy_index):
    '''
    Computes the radius of gyration (Rg) for a set of genomic regions across 
    a whole population of structures.

    Note that in case of multiple copies of a chromosome, a selection
    of the beads corresponding to the regions is performed (see `Notes`
    below.

    Parameters
    ----------
    crd (np.array) : input coordinates (n_structures x n_beads x 3)
    cluster (list of ints) : the ids of regions in the set
    index (alabtools.Index) : the index for the system
    copy_index (list of list) : the bead indexes corresponding to
        each region in the system. In a diploid genome, each region
        maps to 2 beads, in a quadruploid genome to 4 beads, etc.

    Returns
    -------
    rg2s (np.array) : the squared radius of gyration for each structure
    best_structure (int) : the index of the structure which minimize
        the Rg, i.e. rg2s[best_structure] = min(rg2s)

    Notes
    -----
    Since the cell could have multiple copies of the same
    chromosome, we want to select the combination of copies
    which, being more compact, yields the minimum Rg
    The combinatorial problem is large. If there are N
    regions we should check all the possible combinations.
    To make it faster, we group the regions by chromosome.
    Each of these groups, we assume, will come from the same
    copy, not from different ones.
    This way, we have to explore the combinations of groups,
    instead of all the combinations of regions. While regions
    can be hundreds, groups are surely less than the number
    of chromosomes.
    We thus arbitrarily select a representative region from each group,
    and find the optimal copy selection (we call it copy vector:
    if there are two groups {chr1, chr2}, and the minimum Rg is obtained 
    when considering the second copy for chr1 and the first for chr2,
    the copy vector will look like [1, 0]).
    We find a copy vector for each structure, and then use this 
    selection for computing the radius of gyration for all the
    regions.
    '''

    # Group by regions
    n_regions = len(cluster)
    all_chroms = set(index.chrom)
    n_struct = crd.shape[0]
    regions_by_chrom = {c: list() for c in all_chroms}
    for i in cluster:
        c = index.chrom[i]
        regions_by_chrom[c].append(i)

    # Select representatives
    representative_regions = [x[0] for x in regions_by_chrom.values() if len(x)]
    representative_beads = []
    for i in representative_regions:
        representative_beads += copy_index[i]

    rep_copies_num = np.array([len(copy_index[i]) 
                              for i in representative_regions], dtype='i4')
    
    # Select best combinations of representatives
    rep_crd = crd[:, representative_beads, :].astype('f4')
    rep_crd = np.asarray(rep_crd, order='C')
    rrg, rmin, rep_copy_vec = get_rgs2(rep_crd, rep_copies_num)
    
    # compute rg for the full regions by using the copy selection at the 
    # previous step

    full_crd = np.empty((n_struct, n_regions, 3), dtype='f4')
    full_crd = np.asarray(full_crd, order='C')
    for s in xrange(n_struct):
        curr_copy_vec = rep_copy_vec[s]
        selected_ii = []
        for ic, regions in enumerate(regions_by_chrom.values()):
            copy_num = curr_copy_vec[ic]
            for i in regions:
                selected_ii.append(copy_index[i][copy_num])
        full_crd[s, :] = crd[s, selected_ii]

    # we alread chose the copy, so we set the number of copies 
    # for each region to 1
    full_copy_num = np.array([1] * n_regions, dtype='i4')

    rg2s, best_structure, _ = get_rgs2(full_crd, full_copy_num)

    return rg2s, best_structure
    





