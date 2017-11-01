import numpy as np
import h5py
import os
import logging
import time

from .parallel_controller import ParallelController
from sprite_assignment import compute_gyration_radius
from .population_coords import TransposedMatAcc
from alabtools.analysis import HssFile
from .util import pretty_tdelta
            
class Assigner(object):
    def __init__(self, n_tot_clusters, n_struct):
        self.n_struct = n_struct
        self.occupancy = np.zeros(n_struct, dtype=np.int32)
        self.n_tot_clusters = n_tot_clusters
        self.assignment = np.zeros(n_tot_clusters, dtype=np.int32)
        self.aveN = float(n_tot_clusters) / n_struct
        self.stdN = np.sqrt(self.aveN)
        
    def assign_clusters(self, istart, results):
        for i, (best_rgs, curr_idx) in enumerate(results):
            ci = i + istart # cluster id
            penalizations = np.clip(self.occupancy[curr_idx] - self.aveN, 0., None)/self.stdN
            kT = (best_rgs[1] - best_rgs[0]) /  3
            E = (best_rgs-best_rgs[0])/kT + penalizations
            P = np.cumsum(np.exp(-(E-E[0])))
            Z = P[-1]
            e = np.random.rand()*Z
            pos = np.searchsorted(P, e, side='left')
            si = curr_idx[pos]
            self.assignment[ci] = si
            self.occupancy[si] += 1
            
def run_rgs_job(batch_id, batch_size, hss_input, crdinput, cluster_file, dbfile, name, keep_best=100):
    idx_file = 'modeling/SPRITE/%s/tmp.%d.idx.npy' % (name, batch_id)
    val_file = 'modeling/SPRITE/%s/tmp.%d.values.npy' % (name, batch_id)
    
    with HssFile(hss_input, 'r') as f:
        index = f.index
    
    crd = TransposedMatAcc(crdinput)
    cstart = batch_id * batch_size
    cstop = (batch_id + 1) * batch_size
    with h5py.File(cluster_file) as f:
        ii = f['indptr'][cstart:cstop + 1][()]
        clusters = [f['data'][ii[j]:ii[j + 1]] for j in range(len(ii)-1)]

    indexes = []
    values = []
    for cluster in clusters:
        rg2s, _ = compute_gyration_radius(crd, cluster, index, index.copy_index)
        ind = np.argpartition(rg2s, keep_best)[:keep_best] # this is O(n_struct)
        ind = ind[np.argsort(rg2s[ind])] # sorting is O(keep_best ln(keep_best))
        best_rg2s = rg2s[ind]
        indexes.append(ind)
        values.append(best_rg2s)

    np.save(val_file, values)
    np.save(idx_file, np.array(indexes, dtype=np.int32))

    # verify pickle integrity
    values = np.load(val_file)
    indexes = np.load(idx_file)
            
    return 0

def create_cluster_file(fname, clusters, assignment=None):
    data = np.concatenate(clusters)
    indptr = np.cumsum([0] + [len(c) for c in clusters])
    with h5py.File(fname, 'w') as f:
        f.create_dataset('data', data=data, dtype=np.int32)
        f.create_dataset('indptr', data=indptr, dtype=np.int32)
        if assignment is not None:
            f.create_dataset('assignment', data=assignment, dtype=np.int32)
    

def sprite_astep(name, clusters, status_db, hss, crdinput, n_clusters_per_job=250):
    
    logger = logging.getLogger('SPRITE/%s' % name)
        
    os.system('mkdir -p modeling/SPRITE/%s' % name)
    cluster_file = 'modeling/SPRITE/assignment-%s.h5' % name
    tmp_cluster_file = 'modeling/SPRITE/%s/tmp.clusters.h5' % name
    
    if not os.path.isfile(tmp_cluster_file):
        create_cluster_file(tmp_cluster_file, clusters)
    
    to_process = range(len(clusters) // n_clusters_per_job + 1)
    
    pc = ParallelController(name='SPRITE/%s' % name, serial_fun=run_rgs_job, 
                            args=to_process, 
                            max_batch=2000, max_exec_time=180, dbfile=status_db)

    pc.set_const('batch_size', n_clusters_per_job)              
    pc.set_const('hss_input', hss)
    pc.set_const('crdinput', crdinput)
    pc.set_const('cluster_file', tmp_cluster_file)
    pc.set_const('dbfile', status_db)
    pc.set_const('keep_best', 100)
    pc.set_const('name', name)
    
    pc.submit()
    
    if not pc.success():
        logger.error('Was not able to complete the parallel run')
        raise RuntimeError('Was not able to complete the parallel run')
        
    tstart = time.time()
    logger.info('Radius of gyration calculation done, assigning clusters')
    with HssFile(hss) as f:
        n_struct = f.nstruct

    assigner = Assigner(len(clusters), n_struct)
    random_order = np.random.permutation(to_process)
    for i in random_order:
        indexes = np.load('modeling/SPRITE/%s/tmp.%d.idx.npy' % (name, i))
        values = np.load('modeling/SPRITE/%s/tmp.%d.values.npy' % (name, i))
        assigner.assign_clusters(i*n_clusters_per_job, zip(values, indexes))
        
    create_cluster_file(cluster_file, clusters, assigner.assignment)
    
    logger.info('Assignment done. Elapsed: %s', pretty_tdelta(time.time() - tstart))
        
    # cleanup
    os.remove(tmp_cluster_file)
    for i in to_process:
        os.remove('modeling/SPRITE/%s/tmp.%d.idx.npy' % (name, i))
        os.remove('modeling/SPRITE/%s/tmp.%d.values.npy' % (name, i))
    os.rmdir('modeling/SPRITE/%s' % name)
    return assigner.assignment