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
from numpy.lib.format import open_memmap
import time
import h5py
import gzip
from bisect import bisect
from itertools import islice

from alabtools.utils import HssFile




__author__  = "Guido Polles"
__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "polles@usc.edu"

# specifies the size (in bytes) for chunking the process
chunk_size_hint = 500 * 1024 * 1024
actdist_shape = [('i', 'int32'), ('j', 'int32'), ('ad', 'float32'), ('plast', 'float32')]


def _get_copy_index(index):
    tmp_index = {}
    for i, (chrom, start, end, _) in enumerate(index):
        if (chrom, start, end) not in tmp_index:
            tmp_index[(chrom, start, end)] = [i]
        else:
            tmp_index[(chrom, start, end)] += [i]
    # use first index as a key
    copy_index = {ii[0]: ii for ii in tmp_index}
    return copy_index

def get_sorted_coo_matrix(matrix_file, out_matrix_file, chunksize=int(1e7)):
    with h5py.File(matrix_file, 'r') as h5f:
        #  TODO: here I should use an extern sort to deal with very large 
        #  matrices
        nnz = h5f['/matrix/data'].shape[0]
        vals = h5f['/matrix/data'][()]
        sorted_pos = np.argsort(vals)

        adf = open_memmap(out_matrix_file, 'w+', shape=(nnz,), 
                          dtype=[('i', 'int32'), ('j', 'int32'), ('p', 'float32')])

        indptr = h5f['/matrix/indptr'][()]

        ii = np.searchsorted(indptr, sorted_pos, 'right')
        for k, pos in enumerate(sorted_pos):
            i = ii[k]
            j = h5f['/matrix/indices'][pos][()]
            v = vals[pos]
            adf[k] = (i, j, v)
        adf.flush()

def _load_memmap(name, path, mode='r+', shape=None, dtype='float32'):
    """load a file on disk into the interactive namespace as a memmapped array"""
    from numpy.lib.format import open_memmap
    globals()[name] = open_memmap(path, mode=mode, shape=shape, dtype=dtype)

def _compute_actdist(i, j, pwish, plast):
    '''
    Serial function to compute the activation distances for a pair of loci 
    It expects some variables to be defined in its scope:
        'coordinates': a numpy-like array of the coordinates.
        'radii': a numpy-like array of bead radii
        'copy_index': a dict specifying all the copies of each locus

    Parameters
    ----------
        i (int): index of the first locus
        j (int): index of the second locus
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

    global coordinates
    global radii
    global copy_index

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
    ii = copy_index[i]
    jj = copy_index[j]

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

def get_actdists(parallel_client, hss_fname, matrix_memmap, crd_memmap, 
                 actdist_file, new_actdist_file, theta, 
                 max_round_size=int(1e7), chunksize=30):
    '''
    Compute activation distances using ipyparallel. 

    Parameters
    ----------
        parallel_client : ipyparallel.Client
            an ipyparallel Client instance to send parallel jobs to
        hss_fname : str 
            an hss filename containing radii an index
        matrix_memmap : str 
            the contact probability matrix file
        crd_memmap : str 
            the coordinates file
        actdist_file : str 
            last activation distances file.
        new_actdist_file : str 
            file where to save the newly computed activation distances
        theta : float
            consider only contacts with probability greater or equal to theta
        max_round_size : int
            Maximum number of items processed at one time
        chunksize : int
            Number of items in a single request to a worker

    Returns
    -------
        None 

    Raises
    ------
        RuntimeError : if the parallel client has no registered workers.
    '''

    # setup logger
    logger = logging.getLogger()
    
    # check that we have workers ready
    engine_ids = list(parallel_client.ids)
    n_workers = len(engine_ids)
    if n_workers == 0:
        logger.error('get_actdists(): No Engines Registered')
        raise RuntimeError('get_actdists(): No Engines Registered')
    
    logger.info('Starting contact activation distance job on %s, p = %.3f,'
                ' %d workers', crd_memmap, theta, n_workers)

    # reads index and radii from the hss file
    with HssFile(hss_fname) as hss:
        radii = hss.get_radii()
        index = hss.get_index()

    # generate an index of the copies of each locus
    copy_index = _get_copy_index(index)

    # Send static data and open the memmap of the coordinates on workers
    workers = parallel_client[engine_ids]
    workers.apply_sync(_load_memmap, 'coordinates', crd_memmap, mode='r')
    workers['radii'] = radii
    workers['copy_index'] = copy_index
    
    # opens actdists files
    ad_in = gzip.open(actdist_file)
    ad_out = gzip.open(new_actdist_file, 'w')

    # opens probability matrix file
    pm = open_memmap(matrix_memmap, 'r')

    # process matrix
    round_num = 0
    done = False
    while not done:
        round_num += 1
        logger.info('Round number: %d', round_num)
        start = time.time() 

        # get a chunk of the matrix in memory
        c0 = (round_num - 1) * max_round_size
        c1 = round_num * max_round_size
        if c1 >= len(pm):
            c1 = len(pm)
            done = True 
        items = pm[c0:c1][:]

        # discard items < theta
        is_used = items >= theta
        n_used = np.count_nonzero(is_used)
        # matrix is sorted, if there's an element less than theta,
        # all the following elements will be < theta
        if np.size(is_used) - n_used > 0:
            done = True

        # read the last refined probabilities, if present
        old_ads = []
        for line in islice(ad_in, n_used):
            lsp = line.split()
            i = int(lsp[0])
            j = int(lsp[1])
            p = int(lsp[3])
            old_ads.append((i, j, p))

        n_ads = len(old_ads)

        # prepare the arguments
        pargs = []
        for k in range(n_used):
            i, j, p = items[k]
            if k < n_ads:
                assert old_ads[k][0] == i and old_ads[k][1] == j
                op = old_ads[k][2]
            else:
                op = 0
            pargs.append((i, j, p, op))

        # send the jobs to the workers
        workers = parallel_client.load_balanced_view(engine_ids)
        results = workers.map_async(_compute_actdist, pargs, 
                                    chunksize=chunksize)

        for r in results:
            ad_out.write('%d %d %.2f %.4f\n' % r)

        end = time.time()
        logger.debug('get_actdist(): timing for round %d (%d items): %f', round_num,
                     n_used, end-start)

