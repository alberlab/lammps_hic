import numpy as np
from numpy.linalg import norm

from ..actdist import _get_copy_index
from ..lammps_utils import BS, BT


def apply_hic_restraints(system, crd, radii, index, actdists, user_args):
    
    copy_index = _get_copy_index(index)

    for (i, j, pwish, d, p, pnow) in actdists:
        ii = copy_index[i]
        jj = copy_index[j]

        n_combinations = len(ii)*len(jj)
        n_possible_contacts = min(len(ii), len(jj))
        
        cd = np.zeros(n_combinations)
        # NOTE: activation distances are surface to surface, not center to center
        rsph = radii[i] + radii[j]
        dcc = d + rsph  # center to center distance

        it = 0
        for k in ii:
            for m in jj:
                x = crd[k]
                y = crd[m] 
                cd[it] = norm(x - y)
                it += 1

        # find closest distance and remove incompatible contacts
        idx = np.argsort(cd)

        if cd[idx[0]] > dcc:
            continue

        # we have at least one bond to enforce
        parms = {
            'style' : BS.HARMONIC_UPPER_BOUND,
            'k' : user_args['contact_kspring'],
            'r' : user_args['contact_range'] * rsph,
        }

        for it in range(n_possible_contacts):
            if cd[it] > dcc:
                break
            else:
                k = ii[it // len(jj)]
                m = jj[it % len(jj)]
                system.add_bond(system.atoms[k], system.atoms[m], parms, BT.HIC)
                

        