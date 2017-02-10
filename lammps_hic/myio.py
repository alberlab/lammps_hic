from __future__ import print_function, division

import json
import numpy as np
import h5py
import logging
import os
import os.path


ACTDIST_TXT_FMT = '%6d %6d %.5f %10.2f %.5f %.5f'
violations_dtype = np.dtype([('i', 'i4'),
                             ('j', 'i4'),
                             ('absv', 'f4'),
                             ('relv', 'f4'),
                             ('bt', 'S64')])


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
        if i is None:
            crd = f['coordinates'][()]
        else:
            crd = f['coordinates'][i, :, :][()]
        radii = f['radius'][()]
        chrom = f['idx'][()]
        n_struct = f['nstruct'][()]
        n_bead = f['nbead'][()]
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


def read_hms(fname):

    with h5py.File(fname, 'r') as f:
        crd = f['coordinates'][()]
        radii = f['radius'][()]
        chrom = f['idx'][()]
        n_bead = f['nbead'][()]
        violations = f['violations'][()]
        info = json.loads(f['info'][()])
    return crd, radii, chrom, n_bead, violations, info


def write_hms(fname, crd, radii, chrom, violations=[], info={}):
    
    if len(radii) != crd.shape[0]:
        logging.warning('write_hms(): len(radii) != crd.shape[0]')

    if len(chrom) != crd.shape[0]:
        logging.warning('write_hms(): len(chrom) != crd.shape[0]')
    
    with h5py.File(fname, 'w') as f: 
        f.create_dataset('coordinates', data=crd, dtype='f4')
        f.create_dataset('radius', data=radii, dtype='f4')
        f.create_dataset('idx', data=chrom)
        f.create_dataset('nbead', data=crd.shape[0], dtype='i4')
        f.create_dataset('violations', data=np.array(violations, dtype=violations_dtype))
        f.create_dataset('info', data=json.dumps(info))


def pack_hss(prefix, n_struct, out_fname):
    '''
    Create an hss file from multiple hms files
    '''
    crd, radii, chrom, _, _, _ = read_hms('{}_{}.hms'.format(prefix, 0))
    crds = [crd]
    for i in range(n_struct):
        fname = '{}_{}.hms'.format(prefix, i)
        with h5py.File(fname, 'r') as f:
            crds.append(f['coordinates'][()])
    write_hss(out_fname, crds, radii, chrom)


def write_info_file(fname, infos):
    if len(infos) == 0:
        open(fname, 'w')  # writes an empty file and returns
        return
    keys = infos[0].keys()
    with open(fname, 'w') as f:
        print('# structure', file=f, end='  ')
        for k in keys:
            print(str(k), file=f, end='  ')
        for i, info in enumerate(infos):
            for k in keys:
                print(str(info[k]), file=f, end='  ')


def write_violations_file(fname, violations_list):
    with open(fname, 'w') as f: 
        for i, violations in enumerate(violations_list):
            if len(violations) > 0:
                print('STRUCTURE: {}'.format(i), file=f)
                for v in violations:
                    print(v['bt'], ':', v['i'], v['j'], v['absv'], v['relv'])

    

def clear_hms(prefix, n_struct, violations_file=None, info_file=None):
    '''
    Removes hms files.
    If violation_file and/or info_file are specified,
    will save the relative information instead of discarding it.
    On several iterations, the number of hms files becomes very large. 
    Once packed in a hss, it is reasonable to remove them and eventually pack
    the rest of the information on a single file.
    '''
    violations = []
    infos = []
    if violations_file is not None:
        for i in range(n_struct):
            fname = '{}_{}.hms'.format(prefix, i)
            with h5py.File(fname, 'r') as f:
                violations.append(f['violations'][()])
        write_violations_file(violations_file)

    if info_file is not None:    
        for i in range(n_struct):
            with h5py.File(fname, 'r') as f:
                infos.append(json.loads(f['info'][()]))

    for i in range(n_struct):
        os.remove(fname)





