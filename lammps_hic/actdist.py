#!/usr/bin/env python

# Copyright (C) 2016 University of Southern California and
#                        Guido Polles
# 
# Authors: Guido Polles, Nan Hua
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy as np
import scipy.sparse 
import scipy.io
from itertools import izip
import time
import gzip

from .myio import read_hss, read_full_actdist
from .util import monitor_progress, pretty_tdelta


__author__  = "Guido Polles"
__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "polles@usc.edu"

# specifies the size (in bytes) for chunking the process
chunk_size_hint = 500 * 1024 * 1024

def _load_memmap(name, path, mode='r+', shape=None, dtype='float32'):
    """load a file on disk into the interactive namespace as a memmapped array"""
    import numpy as np
    globals()[name] = np.memmap(path, mode=mode, shape=shape, dtype=dtype)


def _compute_actdist(i, j, ii, jj, pwish, plast):
    '''
    Serial function to compute the activation distances for a pair of loci 
    It expects some variables to be defined in its scope:
        'coordinates': a numpy-like array of the coordinates.
        'radii': a numpy-like array of bead radii

    Parameters
    ----------
        i (int): index of the first locus
        j (int): index of the second locus
        ii (list): list of bead indexes corresponding to locus i
        jj (list): list of bead indexes corresponding to locus j
        pwish (float): target contact probability
        plast (float): the last refined probability

    Returns
    -------
        tuple: contains the following items
            - i (int)
            - j (int)
            - ad (float): the activation distance
            - p: the refined probability
    '''

    def existingPortion(v, rsum):
        return sum(v<=rsum)*1.0/len(v)

    def cleanProbability(pij, pexist):
        if pexist < 1:
            pclean = (pij - pexist) / (1.0 - pexist)
        else:
            pclean = pij
        return max(0, pclean)
        

    import numpy as np
    
    n_struct = coordinates.shape[0]

    n_combinations = len(ii)*len(jj)
    n_possible_contacts = min(len(ii), len(jj))

    ri, rj = radii[ii[0]], radii[jj[0]]

    d_sq = np.empty((n_combinations, n_struct))    
    for k in ii:
        for m in jj:
            x = coordinates[:, k, :]
            y = coordinates[:, m, :] 
            d_sq[i] = np.sum(np.square(x - y), axis=1)
    
    pnow = 0
    rcutsq = np.square(2 * (ri + rj))
    d_sq.sort(axis=0)

    #for n in range(n_struct):
    #    z = d_sq[:, n]
        #icopy_selected = [False] * len(ii)
        #jcopy_selected = [False] * len(jj)

        #sorted_indexes = np.argsort(z)

        #for index in sorted_indexes:
        #    icopy = index / len(jj)
        #    jcopy = index % len(jj)
        #    if (icopy_selected[icopy] == False and 
        #        jcopy_selected[jcopy] == False):
        #        icopy_selected[icopy] == True
        #        jcopy_selected[jcopy] == True
        #        selected_dist.append(d_sq[index])
        #        pnow += np.count_nonzero(d_sq[index] <= rcutsq)
    
    contact_count = np.count_nonzero(d_sq[0:n_possible_contacts, :] <= rcutsq)
    pnow = float(contact_count) / (n_possible_contacts * n_struct)
    sortdist_sq = np.sort(d_sq[0:n_possible_contacts, :])

    t = cleanProbability(pnow, plast)
    p = cleanProbability(pwish, t)

    res = None
    if p>0:
        o = min(n_possible_contacts * n_struct - 1, 
                int(round(n_possible_contacts * p * n_struct)))
        activation_distance = np.sqrt(sortdist_sq[o]) - (ri + rj)
        res = (i, j, activation_distance, p)
    return res


def get_actdists(parallel_client, crd_fname, probability_matrix, theta, last_ad, save_to=None, scatter=3):
    '''
    Compute activation distances using ipyparallel. 

    Arguments:
    
        parallel_client (ipyparallel.Client): an ipyparallel Client istance 
            to send jobs
        crd_fname (str): an hss filename of the coordinates
        probability_matrix (str): the contact probability matrix file
        theta (float): consider only contacts with probability greater or 
            equal to theta
        last_ad: last activation distances. Either the filename or an 
            iterable with the activation distances of the last step, or None.
        save_to (str): file where to save the newly computed activation 
            distances
        scatter 
            level of block subdivisions of the matrix. This function
            divide the needed computations into blocks before sending the request
            to parallel workers. It is a compromise between (i) sending coordinates for
            every i, j pair computation and (ii) sending all the coordinates to the workers. 
            Option (i) would require a lot of communication, while option
            (ii) would require a lot of memory on workers.
            Hence, i's and j's are subdivided into blocks and sent to the workers, toghether
            with the needed coordinates. Note that usually blocks on the diagonal 
            have a ton more contacts, hence for load balancing purposes the total number of blocks
            should be larger than the number of workers. scatter=1 means that the 
            number of blocks is just big enough to have all workers receiving at
            least 1 job. Hence, blocks are big and the load is not really balanced.
            Higher values of scatter correspond to smaller blocks and better balancing
            at the cost of increased communication. Note that scatter increase
            the *linear* number of blocks, so scatter=10 means that the total
            number of blocks is ~100 times the number of workers.

    Returns:
        numpy.recarray:
            a numpy recarray with the newly computed activation distances. If
            *save_to* is not None, the recarray will be dumped to the 
            specified file 

    Raises:
        RuntimeError if the parallel client has no registered workers.
    '''

    # setup logger
    logger = logging.getLogger()
    logger.info('Starting contact activation distance job on %s, p = %.3f, %d workers', crd_fname, theta, len(parallel_client))

    # read the matrix in gzipped sparse coordinates format
    fp = gzip.open(probability_matrix)
    lines = fp.readlines(chunk_size_hint)
    while len(lines) != 0:
         


    # transform any matrix argiment in a scipy coo sparse matrix
    start = time.time() 
    if isinstance(probability_matrix, str) or isinstance(probability_matrix, unicode):
        probability_matrix = scipy.io.mmread(probability_matrix)
    elif isinstance(probability_matrix, np.ndarray):
        probability_matrix = scipy.sparse.coo_matrix(np.triu(probability_matrix, 2))
    end = time.time()
    logger.debug('get_actdist(): timing for matrix io: %f', end-start)

    # read the last activation distances and create a dictionary
    if last_ad is None:
        last_ad = []
    elif isinstance(last_ad, str) or isinstance(last_ad, unicode):
        last_ad = read_full_actdist(last_ad)

    # read coordinates from hss  
    start = time.time() 
    crd, radii, chrom, n_struct, n_bead = read_hss(crd_fname)    
    n_loci = probability_matrix.shape[0]
    last_prob = {(i, j): p for i, j, pw, d, p, pn in last_ad}
    n_workers = len(parallel_client.ids)
    end = time.time()
    logger.debug('get_actdist(): timing for dictionary creation: %f', end-start)

    # check we have workers
    if n_workers == 0:
        logger.error('get_actdists(): No Engines Registered')
        raise RuntimeError('get_actdists(): No Engines Registered')
    
    # heuristic to have reasonably small communication but
    # using all the workers. We subdivide in blocks the i, j
    # matrix, such that the blocks are ~10 times the number of 
    # workers. This allows for some balancing but not resulting
    # in eccessive communication.
    start = time.time() 
    blocks_per_line = scatter * int(np.sqrt(0.25 + 2 * n_workers) - 0.5)
    if blocks_per_line > n_loci:
        blocks_per_line = n_loci
    block_size = (n_loci // blocks_per_line) + 1
    blocks = {(i, j): list() for i in range(blocks_per_line) for j in range(i, blocks_per_line)}
    
    for i, j, v in izip(probability_matrix.row, probability_matrix.col, probability_matrix.data):
        if v >= theta:
            if i > j:
                i, j = j, i
            lp = last_prob.get((i, j), 0)
            blocks[(i // block_size, j // block_size)].append((i, j, v, lp))

        
    # select the data chunks to be sent to the workers
    local_data = []
    for k in range(blocks_per_line):
        offset1 = k * block_size
        offset2 = k * block_size
        x1 = crd[:, k * block_size:min((k + 1) * block_size, n_loci), :].transpose((1, 0, 2))
        x2 = crd[:, k * block_size + n_loci:min((k + 1) * block_size + n_loci, 2*n_loci), :].transpose((1, 0, 2))
        local_data.append({'offset1' : offset1, 'offset2': offset2, 'x1': x1, 'x2': x2})
    
    # prepare the arguments. We send a block only if there's work to do
    args = []
    for bi in range(blocks_per_line):
        for bj in range(bi, blocks_per_line):
            if len(blocks[(bi, bj)]) > 0:
                args.append([local_data[bi], local_data[bj], radii, blocks[(bi, bj)], n_struct])
                # logger.debug('get_actdist(): block: %d %d n: %d', bi, bj, len(blocks[(bi, bj)]))
    end = time.time()
    logger.debug('get_actdist(): timing for data distribution: %f', end-start)
    
    # actually send the jobs
    engine_ids = list(parallel_client.ids)
    dview = parallel_client[:]
    dview.use_cloudpickle()
    lbview = parallel_client.load_balanced_view(engine_ids)
    async_results = lbview.map_async(_compute_actdist, args)
    monitor_progress('get_actdists()', async_results)
        
    # merge results
    results = []
    for r in async_results.get():
        results += r
        
    if save_to is not None:
        np.savetxt(save_to,
                   results,
                   fmt='%6d %6d %.5f %10.2f %.5f %.5f')  #myio.ACTDIST_TXT_FMT)

    columns =[('i', int),
              ('j', int),
              ('pwish', float),
              ('actdist', float),
              ('pclean', float),
              ('pnow', float)]

    logger.info('Done with contact activation distance job on %s, p = %.3f (walltime: %s)', crd_fname, theta, pretty_tdelta(async_results.wall_time))
    
    return np.array(results, dtype=columns).view(np.recarray)

