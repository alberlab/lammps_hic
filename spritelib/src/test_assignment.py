from sprite_assignment import compute_gyration_radius, get_rgs2
from alabtools.utils import Index 
import numpy as np

def get_copy_index(index):
    tmp_index = {}
    for i, v in enumerate(index):
        locus = (int(v.chrom), int(v.start), int(v.end))
        if locus not in tmp_index:
            tmp_index[locus] = [i]
        else:
            tmp_index[locus] += [i]
    # use first index as a key
    copy_index = {ii[0]: ii for locus, ii in tmp_index.items()}
    return copy_index

def gyration_radius(crds):
    mu = np.average(crds, axis=0)
    r2 = 0.
    for r in crds:
        r2 += np.dot(r-mu, r-mu)
    r2 /= len(crds)
    return r2

def order_crds(crds, ncs):
    oc = []
    k = 0
    for n in ncs:
        oc.append([crds[k+i] for i in xrange(n)])
        k += n
    return oc

def min_gyr(icrd, ncs):
    crds = order_crds(icrd, ncs)
    n_comb = 1

    for n in ncs: n_comb *= n
    g = []
    for z in xrange(n_comb):
        sel = []
        k = z
        for i in xrange(len(ncs)):
            sel.append(crds[i][k % ncs[i]])
            k //= ncs[i]
        g.append(gyration_radius(np.array(sel)))
    return min(g)

def test1():
    crds = np.random.random((1, 5, 3)).astype('f4')
    ncs = np.array([2, 3], dtype='i4')
    mg = min_gyr(crds[0], ncs)
    rg2, mins, cpi = get_rgs2(crds, ncs)
    assert (np.abs(mg - rg2[0]) < 0.0001 )
    return True

def test2():
    crds = np.random.random((10, 8, 3)).astype('f4')
    ncs = np.array([2, 3, 1, 2], dtype='i4')
    
    gg = []
    for i in range(10):
        gg.append(min_gyr(crds[i], ncs))
    minrg1 = min(gg)
    best1 = gg.index(minrg1)
    gg = np.array(gg, dtype='float32')

    sysindex = Index([0, 0, 1, 1, 1, 2, 3, 3], 
                     [0] * 8, 
                     [100]*8,
                     copy=[0, 1, 0, 1, 2, 0, 0, 1])
    cluster = [0, 2, 5, 6]
    ci = get_copy_index(sysindex)

    rg2s, best2 = compute_gyration_radius(crds, cluster, sysindex, ci)
    assert (best1 == best2)
    assert np.all(np.abs(gg - rg2s) < 0.0001)
    return True
