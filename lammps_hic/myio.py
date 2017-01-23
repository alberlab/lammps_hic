import numpy as np
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