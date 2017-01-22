import h5py
import logging
import numpy as np

from . import io


def monitor_progress(routine, async_results, timeout=60):
    while not async_results.ready():
        logging.info('%s: completed %d of %d tasks. Time elapsed: %s', 
                     routine,
                     async_results.progress,
                     len(async_results),
                     async_results.elapsed)
        async_results.wait(timeout)


def _compute_actdist(crd_fname, nstruct, nbead):
    '''Calculate actdist closure. Returns a function,
    taking as arguments i, j, pwish, plast.
    Using a closure makes sure that even using
    a load balanced view in ipyparallel every
    engine has his own coordinates and radiuses
    dictionaries. Hopefully this minimizes unneeded
    communication. However, a direct view is probably
    better here.'''
    crd = {}
    radius = {}
    
    def existingPortion(v, rsum):
        return sum(v<=rsum)*1.0/len(v)

    def cleanProbability(pij, pexist):
        if pexist < 1:
            pclean = (pij - pexist) / (1.0 - pexist)
        else:
            pclean = pij
        return max(0, pclean)
    
    def get_crd(i):
        try:
            return crd[i], crd[i+nbead], radius[i]
        except KeyError:
            hss = h5py.File(crd_fname, 'r')
            crd[i] = hss['coordinates'][:,i,:][()]
            crd[i + nbead] = hss['coordinates'][:,i + nbead,:][()]
            radius[i] = hss['radius'][i][()]
        return crd[i], crd[i+nbead], radius[i]
        
    def inner(i, j, pwish, plast):
        '''The function to be returned'''
        import numpy as np

        x1, x2, ri = get_crd(i)
        y1, y2, rj = get_crd(j)
        
        d_sq = np.empty( (4, nstruct) )
        d_sq[0] = np.sum(np.square(x1 - y1), axis=1)
        d_sq[1] = np.sum(np.square(x2 - y2), axis=1)
        d_sq[2] = np.sum(np.square(x1 - y2), axis=1)
        d_sq[3] = np.sum(np.square(x2 - y1), axis=1)

        sel_dist = np.empty(nstruct*2)
        pnow = 0
        rcutsq = np.square( 2*(ri+rj) )

        for n in range(nstruct):
            z = d_sq[:, n]

            # assume the closest pair as the first (eventual) contact  
            # and keep the distance of the only other pair which is compatible with it.
            m = np.argsort(z)[0]
            if m == 0 or m == 1:
                sel_dist[2*n:2*n+2] = z[0:2]
                pnow += np.sum( z[0:2] <= rcutsq )    
            else:                   
                sel_dist[2*n:2*n+2] = z[2:4]
                pnow += np.sum( z[2:4] <= rcutsq )    

        pnow = float(pnow) / (2.0 * nstruct)
        sortdist_sq = np.sort(sel_dist)

        t = cleanProbability(pnow, plast)
        p = cleanProbability(pwish,t)

        res = None
        if p>0:
            o = min(2 * nstruct - 1, int(round(2 * p * nstruct)))
            activation_distance = np.sqrt(sortdist_sq[o]) - (ri + rj)
            res = (activation_distance, p, pnow)

        return res
    return inner


def get_actdists(parallel_client, crd_fname, probability_matrix, theta, last_prob, save_to=None):

    with h5py.File(crd_fname, 'r') as hss:
        n_struct = hss['nstruct'][()]
        n_bead = hss['nbead'][()]

    to_process = []

    for i in range(n_bead):
        for j in range(i + 2, n_bead): # skips consecutive beads
            if probability_matrix[i, j] >= theta:
                if last_prob is not None:
                    try:
                        lp = last_prob[(i, j)]
                    except KeyError:
                        lp = 0
                else:
                    lp = 0
                to_process.append((i, j, probability_matrix[i, j], lp))

    dview = parallel_client[:]
    async_results = dview.apply(_compute_actdist(crd_fname,
                                                 n_struct,
                                                 n_bead), # using the closure
                                to_process)
    
    monitor_progress('get_actdists', async_results)
        
    results = async_results.get()

    if save_to is not None:
        np.savetxt(save_to,
                   [i for i in zip(to_process, results)],
                   fmt=io.ACTDIST_TXT_FMT)

    ad = []
    for i, j, pwish, d, pclean, pnow in zip(to_process, results):
        ad.append(i, j, d)
    return ad