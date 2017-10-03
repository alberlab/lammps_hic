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
import scipy.io
import os.path
from .population_coords import PopulationCrdFile
from .parallel_controller import ParallelController
from .network_sqlite import SqliteClient, SqliteServer
from alabtools.utils import HssFile

__author__  = "Guido Polles"
__license__ = "GPL"
__version__ = "0.1.0"
__email__   = "polles@usc.edu"

# specifies the size (in bytes) for chunking the process
actdist_shape = [('i', 'int32'), ('j', 'int32'), ('ad', 'float32'), ('plast', 'float32')]
actdist_fmt_str = '%6d %6d %10.2f %.5f'

def read_actdist_file(filename):
    if os.path.getsize(filename) > 0:
        ad = np.genfromtxt(filename, dtype=actdist_shape)
        if len(ad.shape) == 0:
            ad = [ad]
        ad = ad.view(np.recarray)
    else:
        ad = []
    return ad

def get_copy_index(index):
    tmp_index = {}
    for i, v in enumerate(index):
        locus = (int(v.chrom), int(v.start), int(v.end))
        if locus not in tmp_index:
            tmp_index[locus] = [i]
        else:
            tmp_index[locus] += [i]
    # use first index as a key
    copy_index = {ii[0]: ii for locus, ii in tmp_index.items()}
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

def existingPortion(v, rsum):
        return sum(v<=rsum)*1.0/len(v)

def cleanProbability(pij, pexist):
    if pexist < 1:
        pclean = (pij - pexist) / (1.0 - pexist)
    else:
        pclean = pij
    return max(0, pclean)

def get_actdist(i, j, pwish, plast, coordinates, radii, copy_index):
    '''
    Serial function to compute the activation distances for a pair of loci 
    It expects some variables to be defined in its scope:
        
    Parameters
    ----------
        i (int): index of the first locus
        j (int): index of the second locus
        pwish (float): target contact probability
        plast (float): the last refined probability
        coordinates (lammps_hic.population_coords.PopulationCrdFile): binary 
            file containing coordinates.
        radii (numpy.ndarray): a numpy-like array of bead radii
        copy_index (dictionary int -> list): a dict specifying the indices of
            all the copies of each locus

    Returns
    -------
        i (int)
        j (int)
        ad (float): the activation distance
        p (float): the corrected probability
    '''
    import numpy as np
    
    n_struct = coordinates.nstruct
    ii = copy_index[i]
    jj = copy_index[j]

    n_combinations = len(ii)*len(jj)
    n_possible_contacts = min(len(ii), len(jj))

    ri, rj = radii[ii[0]], radii[jj[0]]

    d_sq = np.empty((n_combinations, n_struct))  
    it = 0  
    for k in ii:
        for m in jj:
            x = coordinates.get_bead(k)
            y = coordinates.get_bead(m) 
            d_sq[it] = np.sum(np.square(x - y), axis=1)
            it += 1
    
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
    sortdist_sq = np.sort(d_sq[0:n_possible_contacts, :].ravel())

    t = cleanProbability(pnow, plast)
    p = cleanProbability(pwish, t)

    res = None
    if p>0:
        o = min(n_possible_contacts * n_struct - 1, 
                int(round(n_possible_contacts * p * n_struct)))
        activation_distance = np.sqrt(sortdist_sq[o]) - (ri + rj)
        res = (i, j, activation_distance, p)
    return res

def _setup_and_run_task(argtuple, input_hss, input_crd, status_db, max_memory):
    '''
    The serial function to be mapped by ipyparallel. Processes a batch of 
    jobs and save results in a database.

    Parameters
    ----------
    argtuple (tuple): contains the batch ID and the arguments for the batch
    input_hss (str): a hss file containing the index for the system
    input_crd (str): the binary input coordinates file
    status_db (str): a database file where to store completed batches
    max_memory (str or int): maximum memory used for data by the binary 
        coordinates manager

    Returns
    -------
    results (list): list containing all the valid results (not None) from 
        `get_actdist` applied to each parameter of the batch.
    '''

    # connects to the status db server
    status = SqliteClient(status_db)

    batch_id, params = argtuple

    # read index and radii
    with HssFile(input_hss) as hss:
        radii = hss.get_radii()
        index = hss.get_index()

    # get the copy index from the index
    copy_index = get_copy_index(index)

    # opens the coordinates file and loops through the jobs in the batch
    results = []
    with PopulationCrdFile(input_crd, 'r', max_memory=max_memory) as crd:
        for i, j, pwish, plast in params:
            r = get_actdist(i, j, pwish, plast, crd, radii, copy_index)
            if r is not None:
                results.append(r)

    # save the results as text 
    status.execute('INSERT INTO completed VALUES (?, ?, ?, ?, ?)', 
                   (batch_id, int(time.time()), len(params), len(results), 
                    '\n'.join([actdist_fmt_str % x for x in results])))

    return results

def compute_activation_distances(run_name, input_hss, input_crd, input_matrix,
                                 sigma, output_file, last_ad=[], log=None, 
                                 args_batch_size=100, task_batch_size=2000, 
                                 ignore_restart=False, max_memory='2GB'):
    '''
    Computes the activation distances for HiC restraints, and save them in a
    text file.

    Parameters
    ----------
    run_name (str): ID for the run, database file names will be based on this
    input_hss (str): hss file with the index for the system (coordinates will
        be ignored)
    input_crd (str): binary coordinates file
    input_matrix (str): a saved scipy.sparse.coo_matrix representing the
        pairwise contact probabilities.
    sigma (float): compute activation distances only for contact probabilities
        greater or equal than sigma
    output_file (str): output text file written on successful completion
    last_ad (list, optional): the output, including corrected probabilities
        from a previous run. Needed for iterative correction
    log (str, optional): save output and run information on this file
    args_batch_size (int, default=100): number of pairs to be processed
        serially in one batch
    task_batch_size (int, default=2000): split the serial tasks in batches.
        This seems necessary to keep the communication and memory usage
        limited and avoid stalling. Each batch will be run on a sub-process
        to ensure memory is freed after running (apparently there some memory
        leak internal to ipyparallel)
    ignore_restart (bool, default=False): if set to `True`, will just over-
        write previously generated files. The default behavior is to not 
        re-run previously successfully completed jobs batches.
    max_memory (str or int, default='2GB'): sets a limit for the RAM memory
        usage of the coordinates manager on the worker nodes. Useful for 
        nodes with limited resources. Note that is a limit for coordinates
        data only, and does not consider the memory required by running
        the engine or the scripts.

    Returns
    -------
    None

    Outputs
    -------
    output_file: a text file with the results
    output_db: a sqlite3 database file with partial results

    Raises
    ------
    RuntimeError: if any of the parallel tasks fails, the parallel controller
        will raise a runtime error with some information on the failure.

    '''

    logger = logging.getLogger('HiC-Actdist')
    logger.setLevel(logging.DEBUG)

    # read input map
    probability_matrix = scipy.io.mmread(input_matrix)
    mask = probability_matrix.data >= sigma
    ii = probability_matrix.row[mask]
    jj = probability_matrix.col[mask]
    pwish = probability_matrix.data[mask]

    # get last probabilities 
    last_prob = {(i, j): p for i, j, _,  p in last_ad}

    # split jobs arguments in batches
    n_args_batches = len(ii) // args_batch_size
    parallel_args = []
    if len(ii) % args_batch_size != 0:
        n_args_batches += 1
    for b in range(n_args_batches):
        start = b * args_batch_size
        end = min((b+1) * args_batch_size, len(ii))
        params = [(ii[k], jj[k], pwish[k], last_prob.get((ii[k], jj[k]), 0.))
                for k in range(start, end)]
        arg = (b, params)
        parallel_args.append(arg)   

    # Decide file names
    status_db = run_name + '.status.db'

    pc = ParallelController(name=run_name, 
                            logfile=log, 
                            serial_fun=_setup_and_run_task,
                            max_batch=task_batch_size)

    # set the variables on the workers
    pc.set_const('input_hss', input_hss)
    pc.set_const('input_crd', input_crd)
    pc.set_const('status_db', status_db)
    pc.set_const('max_memory', max_memory)
    
    # prepare i/o servers
    status_setup = ('CREATE TABLE completed (batch INT, timestamp INT, '
                    'batchsize INT, resultsize INT, data TEXT)')

    if ignore_restart:
        logger.info('Ignoring completed runs.')
        openmode = 'w'
    else:
        openmode = 'r+'

    with SqliteServer(status_db, status_setup, 
                      mode=openmode, port=48000+np.random.randint(1000)):

        # check if we have already completed tasks
        with SqliteClient(status_db) as clt:
            completed = clt.fetchall('SELECT batch from completed')
        to_process = [parallel_args[x] for x in range(n_args_batches) if x not in completed]
        logger.info('%d jobs completed, %d jobs to be submitted.', 
                    len(completed), len(to_process))

        # map the jobs to the workers
        pc.args = to_process
        pc.submit()

        if pc.success():
            logger.info('Job %s completed', run_name)
            with SqliteClient(status_db) as clt:
                completed = clt.fetchall('SELECT data FROM completed ORDER BY batch')
            outtxt = '\n'.join([c[0] for c in completed])
            with open(output_file, 'w') as outf:
                outf.write(outtxt)



