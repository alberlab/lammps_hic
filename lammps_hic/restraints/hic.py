import numpy as np
from numpy.linalg import norm
from ..lammps_utils import Bond, HarmonicUpperBound

def read_actdists(ad):
    '''
    Reads activation distances.

    Activation distances are a list/array of records.
    Every record consists of:

    - i (int): first bead index

    - j (int): second bead index

    - pwish (float): the desired contact probability

    - actdist (float): the activation distance

    - pclean (float): the corrected probability from the iterative correction

    - pnow (float): the current contact probability
    '''
    assert(ad is not None)
    columns =[('i', int),
              ('j', int),
              #('pwish', float),
              ('actdist', float),
              ('pclean', float),
              ]
              #('pnow', float)]
    actdists = []
    if isinstance(ad, str) or isinstance(ad, unicode):
        ad = np.genfromtxt(ad, dtype=columns)
        if ad.shape == ():
            ad = np.array([ad])
    else:
        actdists = ad
    return actdists


def apply_hic_restraints(model, crd, radii, index, user_args):
    ad_fname = user_args['actdist']
    ck = user_args['contact_kspring']
    crange = user_args['contact_range']
    copy_index = index.copy_index

    actdists = read_actdists(ad_fname)

    for (i, j, d, p) in actdists:
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
        bt = HarmonicUpperBound(k=ck,
                                r0=crange * rsph)
        
        for it in range(n_possible_contacts):
            if cd[idx[it]] > dcc:
                break
            else:
                k = ii[idx[it] // len(jj)]
                m = jj[idx[it] % len(jj)]
                model.add_bond(k, m, bt, Bond.HIC)
                

        