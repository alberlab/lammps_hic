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
    if len(radii) != len(crd[0]):
        logging.warning('write_hss(): len(radii) != crd.shape[1]')

    if len(chrom) != len(radii):
        logging.warning('write_hss(): len(chrom) != len(radii)')
    
    with h5py.File(fname, 'w') as f: 
        f.create_dataset('coordinates', data=crd, dtype='f4')
        f.create_dataset('radius', data=radii, dtype='f4')
        f.create_dataset('idx', data=chrom)
        f.create_dataset('nstruct', data=len(crd), dtype='i4')
        f.create_dataset('nbead', data=len(crd[0]), dtype='i4')


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
    
    if len(radii) != len(crd):
        logging.warning('write_hms(): len(radii) != crd.shape[0]')

    if len(chrom) != len(crd):
        logging.warning('write_hms(): len(chrom) != crd.shape[0]')
    
    with h5py.File(fname, 'w') as f: 
        f.create_dataset('coordinates', data=crd, dtype='f4')
        f.create_dataset('radius', data=radii, dtype='f4')
        f.create_dataset('idx', data=chrom)
        f.create_dataset('nbead', data=crd.shape[0], dtype='i4')
        f.create_dataset('violations', data=np.array(violations, dtype=violations_dtype))
        f.create_dataset('info', data=json.dumps(info))


def remove_hms(prefix, n_struct):
    '''
    Removes hms files.
    '''
    for i in range(n_struct):
        fname = '{}_{}.hms'.format(prefix, i)
        if os.path.isfile(fname):
            os.remove(fname)


def pack_hms(prefix, n_struct, hss=None, violations=None, info=None, remove_after=False):
    try:
        crd, radii, chrom, n_beads, cviol, cinfo = read_hms('{}_{}.hms'.format(prefix, 0))
        
        def _vprint(file, violations, structure_index):
            if len(violations) > 0:
                print('STRUCTURE:', structure_index, file=file)
                for v in violations:
                    print(v['bt'], ':', v['i'], v['j'], v['absv'], v['relv'], file=file)

        def _iprint(file, info, structure_index):
            print(structure_index, file=file, end='  ')
            for k, v in info.items():
                print(v, file=file, end='  ')
            print('', file=file)

        # opens files
        if hss is not None:
            hss_file = h5py.File(hss, 'w')
            hss_file.create_dataset('coordinates', shape=(n_struct, n_beads, 3), dtype='f4')
            hss_file.create_dataset('radius', data=radii, dtype='f4')
            hss_file.create_dataset('idx', data=chrom)
            hss_file.create_dataset('nstruct', data=n_struct, dtype='i4')
            hss_file.create_dataset('nbead', data=n_beads, dtype='i4')
            hss_file['coordinates'][0] = crd

        if violations is not None:
            violations_file = open(violations, 'w')
            _vprint(violations_file, cviol, 0)

        if info is not None:
            info_file = open(info, 'w')

            # print headers
            print('# structure', file=info_file, end='  ')
            for k in cinfo.keys():
                print(str(k), file=info_file, end='  ')
            print('', file=info_file)

            # print first record
            _iprint(info_file, cinfo, 0)

        for i in range(1, n_struct):
            fname = '{}_{}.hms'.format(prefix, i)
        
            with h5py.File(fname, 'r') as f:
                if hss:
                    hss_file['coordinates'][i] = f['coordinates'][()]

                if violations:
                    _vprint(violations_file, f['violations'][()], i)

                if info:
                    _iprint(info_file, json.loads(f['info'][()]), i)

        if hss:
            hss_file.close()
        if violations:
            violations_file.close()
        if info:
            info_file.close()

        if remove_after:
            remove_hms(prefix, n_struct)

    except: # let's make sure the hdf5 file is removed, else weird stuff happens
        if os.path.isfile(hss):
            os.remove(hss)
        raise
    






