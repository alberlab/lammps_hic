import numpy as np
from numpy.linalg import norm
import h5py
from ..lammps_utils import Bond, HarmonicUpperBound, HarmonicLowerBound



def apply_fish_restraints(model, crd, radii, index, user_args):
    
    copy_index = index.copy_index

    hff = h5py.File(user_args['fish'], 'r')
    minradial = 'r' in user_args['fish_type']
    maxradial = 'R' in user_args['fish_type']
    minpair = 'p' in user_args['fish_type']
    maxpair = 'P' in user_args['fish_type']

    struct_id = user_args['i']

    ck = user_args['fish_kspring']
    tol = user_args['fish_tol']
    
    if minradial:
        probes = hff['probes']
        minradial_dist = hff['radial_min'][:, struct_id]
        for k, i in enumerate(probes):
            ii = copy_index[i]  # get all the copies of locus i
            cd = np.zeros(len(ii))
            td = minradial_dist[k]
            for m, x in ii:
                cd[m] = norm(crd[x])
            min_idx = ii[np.argsort(cd)[0]]

            bt = HarmonicLowerBound(k=ck,
                                    r0=max(0, td - tol))            
            center = model.get_next_dummy()
            model.add_bond(min_idx, center, bt, restraint_type=Bond.FISH_RADIAL)

            bt = HarmonicUpperBound(k=ck,
                                    r0=td + tol)
            center = model.get_next_dummy()
            model.add_bond(min_idx, center, bt, restraint_type=Bond.FISH_RADIAL)

    if maxradial:
        cd = np.zeros(2)
        probes = hff['probes']
        maxradial_dist = hff['radial_max'][:, struct_id]
        for k, i in enumerate(probes):
            ii = copy_index[i]  # get all the copies of locus i
            cd = np.zeros(len(ii))
            td = maxradial_dist[k]
            for m, x in ii:
                cd[m] = norm(crd[x])
            max_idx = ii[np.argsort(cd)[-1]]

            bt = HarmonicUpperBound(k=ck,
                                    r0=td + tol)
            center = model.get_next_dummy()
            model.add_bond(max_idx, center, bt, restraint_type=Bond.FISH_RADIAL)

            bt = HarmonicLowerBound(k=ck,
                                    r0=max(0, td - tol))            
            center = model.get_next_dummy()
            model.add_bond(max_idx, center, bt, restraint_type=Bond.FISH_RADIAL)

    if minpair:
        pairs = hff['pairs']
        minpair_dist = hff['pair_min'][:, struct_id]
        for k, (i, j) in enumerate(pairs):
            assert (i != j)
            ii = copy_index[i]
            jj = copy_index[j]

            n_combinations = len(ii)*len(jj)
            
            dmin = minpair_dist[k]

            bt = HarmonicLowerBound(k=ck,
                                    r0=max(0, dmin - tol))
            cd = np.zeros(n_combinations)
            it = 0
            for m in ii:
                for n in jj:
                    x = crd[m]
                    y = crd[n] 
                    cd[it] = norm(x - y)
                    model.add_bond(m, n, bt, restraint_type=Bond.FISH_PAIR)
                    it += 1

            bt = HarmonicUpperBound(k=ck,
                                    r0=dmin + tol)
        
            midx = np.argsort(cd)[0]

            m = midx // len(jj)
            n = midx % len(jj)
            model.add_bond(m, n, bt, restraint_type=Bond.FISH_PAIR)

    if maxpair:
        pairs = hff['pairs']
        maxpair_dist = hff['pair_max'][:, struct_id]
        for k, (i, j) in enumerate(pairs):
            assert (i != j)
            ii = copy_index[i]
            jj = copy_index[j]

            n_combinations = len(ii)*len(jj)

            dmax = maxpair_dist[k]
            
            bt = HarmonicUpperBound(k=ck,
                                    r0=dmax + tol)
            cd = np.zeros(n_combinations)
            it = 0
            for m in ii:
                for n in jj:
                    x = crd[m]
                    y = crd[n] 
                    cd[it] = norm(x - y)
                    model.add_bond(m, n, bt, restraint_type=Bond.FISH_PAIR)
                    it += 1

            bt = HarmonicLowerBound(k=ck,
                                    r0=max(0, dmax - tol))
            midx = np.argsort(cd)[-1]  # get the max distance

            m = midx // len(jj)
            n = midx % len(jj)
            model.add_bond(m, n, bt, restraint_type=Bond.FISH_PAIR)

    hff.close()