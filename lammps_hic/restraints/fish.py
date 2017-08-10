import numpy as np
from numpy.linalg import norm
import h5py

from ..actdist import _get_copy_index
from ..lammps_utils import BS, BT



def apply_fish_restraints(system, crd, index, args):

    copy_index = _get_copy_index(index)

    hff = h5py.File(args['fish'], 'r')
    minradial = 'r' in args['fish_type']
    maxradial = 'R' in args['fish_type']
    minpair = 'p' in args['fish_type']
    maxpair = 'P' in args['fish_type']

    struct_id = args['i']
    
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

            parms = {
                'style' : BS.HARMONIC_LOWER_BOUND,
                'k' : args['fish_kspring'],
                'r' : max(0, td - args['fish_tol']),
            }
            dummy = system.get_next_dummy()
            system.bonds.add_bond(system.atoms[min_idx], dummy, parms, BT.FISH_RADIAL)

            parms = {
                'style' : BS.HARMONIC_UPPER_BOUND,
                'k' : args['fish_kspring'],
                'r' : td + args['fish_tol'],
            }
            dummy = system.get_next_dummy()
            system.bonds.add_bond(system.atoms[min_idx], dummy, parms, BT.FISH_RADIAL)

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

            parms = {
                'style' : BS.HARMONIC_UPPER_BOUND,
                'k' : args['fish_kspring'],
                'r' : td + args['fish_tol'],
            }
            dummy = system.get_next_dummy()
            system.bonds.add_bond(system.atoms[max_idx], dummy, parms, BT.FISH_RADIAL)

            parms = {
                'style' : BS.HARMONIC_LOWER_BOUND,
                'k' : args['fish_kspring'],
                'r' : max(0, td - args['fish_tol']),
            }
            dummy = system.get_next_dummy()
            system.bonds.add_bond(system.atoms[max_idx], dummy, parms, BT.FISH_RADIAL)

    if minpair:
        pairs = hff['pairs']
        minpair_dist = hff['pair_min'][:, struct_id]
        for k, (i, j) in enumerate(pairs):
            assert (i != j)
            ii = copy_index[i]
            jj = copy_index[j]

            n_combinations = len(ii)*len(jj)
            
            dmin = minpair_dist[k]

            parms = {
                'style' : BS.HARMONIC_LOWER_BOUND,
                'k' : args['fish_kspring'],
                'r' : max(0, dmin - args['fish_tol']),
            }
            cd = np.zeros(n_combinations)
            it = 0
            for m in ii:
                for n in jj:
                    x = crd[m]
                    y = crd[n] 
                    cd[it] = norm(x - y)
                    system.add_bond(system.atoms[m], system.atoms[n], 
                                    parms, BT.FISH_PAIR)
                    it += 1

            parms = {
                'style' : BS.HARMONIC_LOWER_BOUND,
                'k' : args['fish_kspring'],
                'r' : dmin + args['fish_tol'],
            }
            
            midx = np.argsort(cd)[0]

            m = midx // len(jj)
            n = midx % len(jj)
            system.add_bond(system.atoms[m], system.atoms[n], 
                            parms, BT.FISH_PAIR)

    if maxpair:
        pairs = hff['pairs']
        maxpair_dist = hff['pair_max'][:, struct_id]
        for k, (i, j) in enumerate(pairs):
            assert (i != j)
            ii = copy_index[i]
            jj = copy_index[j]

            n_combinations = len(ii)*len(jj)

            dmax = maxpair_dist[k]
            
            parms = {
                'style' : BS.HARMONIC_UPPER_BOUND,
                'k' : args['fish_kspring'],
                'r' : dmax + args['fish_tol'],
            }
            cd = np.zeros(n_combinations)
            it = 0
            for m in ii:
                for n in jj:
                    x = crd[m]
                    y = crd[n] 
                    cd[it] = norm(x - y)
                    system.add_bond(system.atoms[m], system.atoms[n], 
                                    parms, BT.FISH_PAIR)
                    it += 1

            parms = {
                'style' : BS.HARMONIC_LOWER_BOUND,
                'k' : args['fish_kspring'],
                'r' : max(0, dmax - args['fish_tol']),
            }

            midx = np.argsort(cd)[-1]  # get the max distance

            m = midx // len(jj)
            n = midx % len(jj)
            system.add_bond(system.atoms[m], system.atoms[n], 
                            parms, BT.FISH_PAIR)

    hff.close()