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


def compute_gyration_radius(crd, cluster, index, copy_index, 
                            full_check=False):
    '''
    Computes the radius of gyration (Rg) for a set of genomic regions across 
    a whole population of structures.

    Note that in case of multiple copies of a chromosome, a selection
    of the beads corresponding to the regions is performed (see `Notes`
    below.

    For performance reasons, the selection is made by randomly selecting only
    one bead per chromosome involved (a `representative`).

    Parameters
    ----------
    crd (np.array) : input coordinates (n_structures x n_beads x 3)
    cluster (list of ints) : the ids of regions in the set
    index (alabtools.Index) : the index for the system
    copy_index (list of list) : the bead indexes corresponding to
        each region in the system. In a diploid genome, each region
        maps to 2 beads, in a quadruploid genome to 4 beads, etc.
    full_check (bool) : whether the final Rg's are computed on the 
        selected representatives only (False, default), or on all the
        cluster beads (True).

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

    cdef int n_regions = len(cluster)
    all_chroms = set(index.chrom)
    cdef int n_struct = crd.shape[0]
    cdef int i
    cdef int c
    
    # Group regions by chromosome
    regions_by_chrom = {c: list() for c in all_chroms}
    for i in cluster:
        c = index.chrom[i]
        regions_by_chrom[c].append(i)

    # Select one representative region for each chromosome                                                          
    cdef np.ndarray[int, ndim=1] representative_regions = np.array(
        [np.random.choice(x) for x in regions_by_chrom.values() if len(x)], dtype='i4')

    # select all the copies for each representative region
    cdef np.ndarray[int, ndim=1] representative_beads = np.array(np.concatenate(
        [copy_index[i] for i in representative_regions]), dtype='i4')

    # count the number of copies for each representative region
    cdef np.ndarray[int, ndim=1] rep_copies_num = np.array(
        [len(copy_index[i]) for i in representative_regions], dtype='i4')

    # hdf5 files require to ask indexes in order
    horder = np.argsort(representative_beads)
    rorder = np.empty(len(representative_beads), dtype=np.int32)
    rorder[horder] = np.arange(len(representative_beads))
    tmp_crd = crd[:, representative_beads[horder], :].astype('f4')

    # rep_crd are the coordinates of all the copies of each representative
    # region, ordered by chromosomes 
    rep_crd = tmp_crd[:, rorder, :]

    rep_crd = np.asarray(rep_crd, order='C')
    cdef np.ndarray[int, ndim=2] rep_copy_vec
    rrg, rmin, rep_copy_vec = get_rgs2(rep_crd, rep_copies_num)
    
    # if we have only the partial check, return here
    if not full_check:
        return rrg, rmin

    # reclaim memory
    del tmp_crd
    del rep_crd

    # compute rg for the full regions by using the copy selection at the 
    # previous step

    # all the regions involved in the clusters, ordered by chromosome
    cdef np.ndarray[int, ndim=1] all_regions = np.array(np.concatenate(
        [x for x in regions_by_chrom.values() if len(x)]), dtype='i4')

    # all the possible beads, i.e. the indexes of all the copies of all 
    # the regions
    cdef np.ndarray[int, ndim=1] all_beads = np.array(np.concatenate(
        [copy_index[i] for i in all_regions]), dtype='i4')
    
    # number of regions for each chromosome
    cdef np.ndarray[int, ndim=1] num_regions_in_chrom = np.array(
        [len(x) for x in regions_by_chrom.values() if len(x)], dtype='i4')

    # mark the positions where regions change chromosome
    cdef np.ndarray[int, ndim=1] chrom_end = np.array(
        np.cumsum(num_regions_in_chrom), dtype='i4')

    # mark the positions of each region in the all_bead vector.
    cdef np.ndarray[int, ndim=1] region_start = np.array(np.cumsum(
        [0] + [len(copy_index[i]) for i in all_regions]), dtype='i4')

    # obtain only the subset of coordinates needed for this computation
    cdef np.ndarray[float, ndim=3] all_crd = crd[:, all_beads, :] 

    # full_crd are the final set of coordinates after copy selection
    cdef np.ndarray[float, ndim=3] full_crd = np.empty((n_struct, n_regions, 3), 
                                                       dtype='f4')
    
    cdef int istruct, ibead, ichrom, iregion 
    cdef np.ndarray[int, ndim=1] curr_copy_vec
    cdef np.ndarray[int, ndim=1] selected_ii = np.zeros(n_regions, dtype='i4')
    for istruct in range(n_struct):
        ichrom = 0 # index of the current chromosome
        for iregion, region in enumerate(all_regions):
            if iregion >= chrom_end[ichrom]:
               ichrom += 1
            # iregion is the index of the region, ibead is the index of the 
            # selected copy in the all_crd array
            ibead = region_start[iregion] + rep_copy_vec[istruct][ichrom]
            selected_ii[iregion] = ibead
        full_crd[istruct, :] = all_crd[istruct][selected_ii, :]

    # we alread chose the copy, so we set the number of copies 
    # for each region to 1
    full_copy_num = np.array([1] * n_regions, dtype='i4')

    # get the squared Rg's, ignore the trivial copy vector 
    rg2s, best_structure, _ = get_rgs2(full_crd, full_copy_num)
    return rg2s, best_structure
    

def assignment(crd, index, clusters, int n_struct, int keep_best=1000, int max_chrom_num=8):
    cdef np.ndarray[int, ndim=1] occupancy = np.zeros(n_struct, dtype=np.int32)
    cdef int n_clusters = len(clusters)
    cdef np.ndarray[int, ndim=1] assignment = np.zeros(n_clusters, dtype=np.int32)
    cdef float aveN = float(n_clusters) / n_struct
    cdef float stdN = np.sqrt(aveN)
    
    cdef float Erange, Emin, kT, Z
    cdef int pos
    cdef int ci = 0
    cdef int max_nrg = keep_best
    for ci, cluster in enumerate(clusters):
        chroms_involved = set()
        for i in cluster:
            chroms_involved.add(index.chrom[i])
        if len(chroms_involved) > max_chrom_num:
            assignment[ci] = -1
            continue
        rg2s, best = compute_gyration_radius(crd, cluster, index, index.copy_index)
        order = np.argsort(rg2s)[:max_nrg]
        best_rgs = rg2s[order]
        penalizations = np.clip(occupancy[order] - aveN, 0., None)/stdN
        Emin = best_rgs[0]
        kT = (best_rgs[1] - best_rgs[0]) /  3
        E = (best_rgs-best_rgs[0])/kT + penalizations
        P = np.cumsum(np.exp(-(E-E[0])))
        Z = P[-1]
        e = np.random.rand()*Z
        pos = np.searchsorted(P, e, side='left')
        si = order[pos]
        assignment[ci] = si
        occupancy[si] += 1
    return assignment



    




