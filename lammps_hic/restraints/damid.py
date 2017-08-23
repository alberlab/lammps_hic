import numpy as np
from numpy.linalg import norm

from ..actdist import _get_copy_index

MAX_BONDS = 20

def apply_damid_restraints(system, crd, radii, index, damid_actdists, user_args):

    copy_index = _get_copy_index(index)
    for item in damid_actdists:
        i = item[0]
        ii = copy_index[i]
        d = item[2]

        # target minimum distance
        td = user_args['nucleus_radius'] - (2 * radii[i])

        # current distances
        cd = np.zeros(len(ii))
        for k in ii:
            cd[k] = user_args['nucleus_radius'] - norm(crd[k])

        parms = None

        for k in range(len(ii)):
            if cd[k] > d:
                continue

            if parms is None:
                parms = {
                    'style' : BS.HARMONIC_LOWER_BOUND,
                    'k' : user_args['damid_kspring'],
                    'r' : td,
                }

            dummy = system.get_next_dummy()

            system.bonds.add_bond(system.atoms[ii[k]], dummy, parms, BT.DAMID)
            
