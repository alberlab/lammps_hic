import numpy as np
import gzip
import multiprocessing
from alabtools.utils import HssFile

r       = None
crd     = None
nbead   = None
nstruct = None

def existingPortion(v, rsum):
    return sum(v<=rsum)*1.0/len(v)

def cleanProbability(pij,pexist):
    if pexist < 1:
        pclean = (pij-pexist)/(1.0-pexist)
    else:
        pclean = pij
    return max(0,pclean)

def calcActdist(v):
    (i, j, chri, chrj, pwish, plast) = v
    global r
    global crd
    global nbead
    global nstruct
    
    d1 = np.linalg.norm(crd[:,i,:] - crd[:,j,:],axis=1) -r[i] - r[j] #L2 norm
    d2 = np.linalg.norm(crd[:,i+nbead,:] - crd[:,j+nbead,:],axis=1)-r[i]-r[j]
    if chri == chrj:
        dists = np.concatenate((d1,d2))
    else:
        d3 = np.linalg.norm(crd[:,i,:] - crd[:,j+nbead,:],axis=1) -r[i] - r[j]
        d4 = np.linalg.norm(crd[:,i+nbead,:] - crd[:,j,:],axis=1) -r[i] - r[j]
        dtable = np.column_stack((d1,d2,d3,d4))
        dists = np.sort(dtable,axis=1)[:,0:2].flatten() #fetch top 2 smallest distance
    #-
    sortdist = np.sort(dists)
    pnow = existingPortion(sortdist,r[i]+r[j])
    
    
    t = cleanProbability(pnow,plast)
    p = cleanProbability(pwish,t)
    

    if p>0:
        o = min(2*nstruct-1, int(round(2*p*nstruct)))
        return '%4d %4d %5.3f %7.1f %5.3f %5.3f'%(i,j,pwish,sortdist[o],p,pnow)
    else:
        return '%4d %4d %5.3f %7.1f %5.3f %5.3f'%(i,j,pwish,0,p,pnow)


def get_actdist(hss, pmat, adin, freq, adout, n_cores=16):
    global crd
    global nbead
    global nstruct
    global r

    if adin is not None:
        adin = gzip.open(adin, 'r')
    to_process = []
    
    with HssFile(hss) as f:
        crd = f.coordinates
        index = f.index
        nbead = f.nbead/2
        nstruct = f.nstruct
        r = f.radii

    for ti, tj, v in pmat:
        i = int(ti)
        j = int(tj)    
        if v >= freq:
            lastp = 0
            if adin is not None:
                line = adin.readline()
                if line:
                    lastp = float(line.split()[4])

            to_process.append((i, j, index[i].chrom, index[j].chrom, v, lastp))
        else:
            break
            # skip low frequency entries

    pool = multiprocessing.Pool(processes=n_cores)
    results = pool.map(calcActdist, to_process)


    ################
    # save results #
    ################

    with gzip.open(adout, 'w') as f:
        for r in results:
            if r is not None:
                f.write(r + '\n')

