import h5py
import logging
import numpy as np

from .myio import read_hss
from .util import monitor_progress



def _compute_actdist(nstruct, nbead):
    '''Calculate actdist closure. Returns a function,
    taking as arguments i, j, pwish, plast.
    Using a closure makes sure that even using
    a load balanced view in ipyparallel every
    engine has his own coordinates and radiuses
    dictionaries. Hopefully this minimizes unneeded
    communication. However, a direct view is probably
    better here.'''
    
    def existingPortion(v, rsum):
        return sum(v<=rsum)*1.0/len(v)

    def cleanProbability(pij, pexist):
        if pexist < 1:
            pclean = (pij - pexist) / (1.0 - pexist)
        else:
            pclean = pij
        return max(0, pclean)
        
    def inner(data):
        '''The function to be returned'''
        import numpy as np
        results = []
        local_i, local_j, radii, to_process = data
        
        for i, j, pwish, plast in to_process:
            
            x1 = local_i['x1'][i-local_i['offset1']] 
            x2 = local_i['x2'][i-local_i['offset2']]
            y1 = local_j['x1'][j-local_j['offset1']] 
            y2 = local_j['x2'][j-local_j['offset2']]
            
            
            ri, rj = radii[i], radii[j]

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
                res = (i, j, pwish, activation_distance, p, pnow)
                results.append(res)
        return results
    
    return inner



def get_actdists(parallel_client, crd_fname, probability_matrix, theta, last_ad, save_to=None):
    
    crd, radii, chrom, n_struct, n_bead = read_hss(crd_fname)    
    n_loci = len(probability_matrix)
    last_prob = {(i, j): p for i, j, pw, d, p, pn in last_ad}
    n_workers = len(parallel_client.ids)
                
    if n_workers == 0:
        logging.error('get_actdists(): No Engines Registered')
        raise RuntimeError('get_actdists(): No Engines Registered')
    
    # heuristic to have reasonably small communication but
    # using all the workers. We subdivide in blocks the i, j
    # matrix, such that the blocks are ~4 times the number of 
    # workers. This allows for some balancing but not resulting
    # in eccessive communication.
    blocks_per_line = 2 * int(np.sqrt(0.25 + 2 * n_workers) - 0.5)
    if blocks_per_line > n_loci:
        blocks_per_line = n_loci
    block_size = (n_loci // blocks_per_line) + 1
    blocks = {(i, j): list() for i in range(blocks_per_line) for j in range(i, blocks_per_line)}
    
    for i in range(n_loci):
        for j in range(i + 2, n_loci): # skips consecutive beads
            if probability_matrix[i, j] >= theta:
                try:
                    lp = last_prob[(i, j)]
                except KeyError:
                    lp = 0    
                blocks[(i // block_size, j // block_size)].append((i, j, probability_matrix[i, j], lp))
        
    local_data = []
    for k in range(blocks_per_line):
        offset1 = k * block_size
        offset2 = k * block_size
        x1 = crd[:, k * block_size:min((k + 1) * block_size, n_loci), :].transpose((1, 0, 2))
        x2 = crd[:, k * block_size + n_loci:min((k + 1) * block_size + n_loci, 2*n_loci), :].transpose((1, 0, 2))
        local_data.append({'offset1' : offset1, 'offset2': offset2, 'x1': x1, 'x2': x2})
    
    args = []
    for bi in range(blocks_per_line):
        for bj in range(bi, blocks_per_line):
            args.append((local_data[bi], local_data[bj], radii, blocks[(bi, bj)]))
    
    lbview = parallel_client.load_balanced_view()
    async_results = lbview.map_async(_compute_actdist(n_struct, n_loci), args)  # using the closure
    
    monitor_progress('get_actdists()', async_results)
        
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
    
    return results

    #return np.array(results).view(np.recarray, dtype=columns)

