import numpy as np
from numpy.linalg import norm
import os.path

from ..actdist import _get_copy_index
from ..lammps_utils import Bond, HarmonicLowerBound


def read_damid(damid):
    assert(damid is not None)
    if isinstance(damid, str) or isinstance(damid, unicode):
        if os.path.getsize(damid) > 0:
            damid_actdists = np.genfromtxt(damid,
                                           usecols=(0, 1, 2),
                                           dtype=(int, float, float))
            if damid_actdists.shape == ():
                damid_actdists = [damid_actdists]
        else:
            damid_actdists = []
    return damid_actdists


def apply_damid_restraints(model, crd, radii, index, user_args):
    
    damid_actdists = read_damid(user_args['damid'])
    copy_index = _get_copy_index(index)
    ck = user_args['damid_kspring']
    
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

        for k in range(len(ii)):
            if cd[k] > d:
                continue
            center = model.get_next_dummy()
            bt = HarmonicLowerBound(k=ck, r0=td)
            model.add_bond(k, center, bt, restraint_type=Bond.DAMID)
            
