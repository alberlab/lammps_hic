import numpy as np
import h5py
import logging
import os.path


ACTDIST_TXT_FMT = '%6d %6d %.5f %10.2f %.5f %.5f'


def read_full_actdist(filename):
    columns =[('i', int),
              ('j', int),
              ('pwish', float),
              ('actdist', float),
              ('pclean', float),
              ('pnow', float)]
    if os.path.getsize(filename) > 0:
        ad = np.genfromtxt(filename, dtype=columns)
        if len(ad.shape) == 0:
            ad = [ad]
        ad = ad.view(np.recarray)
    else:
        ad = []
    return ad


def read_actdist(filename):
    columns =[('i', int),
              ('j', int),
              ('actdist', float),
             ]
    if os.path.getsize(filename) > 0:
        ad = np.genfromtxt(filename, dtype=columns, usecols=(0, 1, 3))
        if len(ad.shape) == 0:
            ad = [ad]
        ad = ad.view(np.recarray)
    else:
        ad = []
    return ad


def write_actdist():
    pass


def read_hss(fname, i=None):
    with h5py.File(fname, 'r') as f:
        if i == None:
            crd = f['coordinates'][:, :, :][()]
        else:
            crd = f['coordinates'][i, :, :][()]
        radii = f['radius'][()]
        chrom = f['idx'][()]
        n_struct = f['nstruct']
        n_bead = f['nbead']
    return crd, radii, chrom, n_struct, n_bead


def write_hss(fname, crd, radii, chrom):
    if len(radii) != crd.shape[1]:
        logging.warning('write_hss(): len(radii) != crd.shape[1]')

    if len(chrom) != crd.shape[1]:
        logging.warning('write_hss(): len(chrom) != crd.shape[1]')
    
    with h5py.File(fname, 'w') as f: 
        f.create_dataset('coordinates', data=crd, dtype='f4')
        f.create_dataset('radius', data=radii, dtype='f4')
        f.create_dataset('idx', data=chrom)
        f.create_dataset('nstruct', data=crd.shape[0], dtype='i4')
        f.create_dataset('nbead', data=crd.shape[1], dtype='i4')
