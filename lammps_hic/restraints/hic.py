import numpy as np
from numpy.linalg import norm

from ..lammps_utils import Bond, HarmonicUpperBound
from ..actdist import _get_copy_index
from ..lammps_utils import BS, BT

def read_actdists(ad):
    assert(ad is not None)
    actdists = []
    if isinstance(ad, str) or isinstance(ad, unicode):
        if ad[-7:] == ".txt.gz":
            import gzip
            with gzip.open(ad) as f:
                for line in f:
                    try:
                        i, j, pwish, d, p, pnow = line.split()
                    except ValueError:
                        continue
                    actdists.append((int(i), int(j), pwish, float(d), p, pnow))
        elif os.path.getsize(ad) > 0:
            actdists = read_full_actdist(ad)
    else:
        actdists = ad
    return actdists


def apply_hic_restraints(model, crd, radii, index, user_args):
    ad_fname = user_args['actdist']
    ck = user_args['contact_kspring']
    crange = user_args['contact_range']
    copy_index = _get_copy_index(index)

    actdists = read_actdists(ad_fname)

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
        bt = HarmonicUpperBound(k=ck,
                                r0=crange * rsph)
        
        for it in range(n_possible_contacts):
            if cd[idx[it]] > dcc:
                break
            else:
                k = ii[idx[it] // len(jj)]
                m = jj[idx[it] % len(jj)]
                model.add_bond(k, m, bt, Bond.HIC)
                

        