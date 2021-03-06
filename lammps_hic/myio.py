#!/usr/bin/env python

# Copyright (C) 2016 University of Southern California and
#                        Guido Polles
# 
# Authors: Guido Polles
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
The **myio** module provides some common IO operations
functions on hms, hss, actdist files.
'''

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
    '''
    Alias for read_full_actdist
    '''
    return read_full_actdist(filename)


def write_actdist():
    pass


def read_hss(fname, i=None):
    '''
    Reads information from an hss file.
    Arguments:
        fname (str): filename of hss file
        i (int): if not None, returns the coordinates of the i-th structure
            instead of the full population

    Returns:
        crd (numpy.ndarray(dtype='f4')): population coordinates
        radii (numpy.ndarray(dtype='f4')): bead radii
        chrom (numpy.ndarray(dtype=str)): chromosome tags
        n_struct (int): number of structures
        n_bead (int): number of beads in one structure
    '''
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
    '''
    Writes an hss file.
    Arguments:
        fname (str): path of hss file
        crd (numpy.ndarray): coordinates of the whole population
        radii (array of floats): radius for each bead
        chrom (array of strings): chromosome tag for each bead
    '''
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
    '''
    Reads an hms file.
    Arguments:
        fname (str): hms file path

    Returns:
        crd (np.ndarray(dtype='f4')): coordinates of the structure
        radii (np.ndarray(dtype='f4')): radius for each bead
        chrom (np.ndarray(dtype='S5')): chromosome tag for each bead
        n_bead (int): the number of beads in the structure
        violations (np.recarray): an array containing the violations.
            Refer to the lammps.check_violations function documentation.
        info (dict): a dictionary containing run informations.
    '''
    with h5py.File(fname, 'r') as f:
        crd = f['coordinates'][()]
        radii = f['radius'][()]
        chrom = f['idx'][()]
        n_bead = f['nbead'][()]
        violations = f['violations'][()]
        info = json.loads(f['info'][()])
    return crd, radii, chrom, n_bead, violations, info


def write_hms(fname, crd, radii, chrom, violations=[], info={}):
    '''
    TODO: write docs
    '''
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


def dump_violations(file, violations, structure_index):
    if len(violations) > 0:
        print('STRUCTURE:', structure_index, file=file)
        for v in violations:
            print(v['bt'], ':', v['i'], v['j'], v['absv'], v['relv'], file=file)


def dump_info_header(file, info):
    print('# structure', file=file, end='  ')
    for k in info.keys():
        print(str(k), file=file, end='  ')
    print('', file=file)


def dump_info(file, info, structure_index):
    print(structure_index, file=file, end='  ')
    for k, v in info.items():
        print(v, file=file, end='  ')
    print('', file=file)


def pack_hms(prefix, n_struct, hss=None, violations=None, info=None, remove_after=False):
    '''
    Compact a series of hms files into single summary files.

    Arguments:
        prefix (str): prefix of hms files. The actual filenames are expected to be in
            the form <prefix>_0.hms, <prefix>_1.hms, ... 
        n_struct (int): number of structure files
        hss (str): if specified, filename where to save coordinates, radii and
            chromosome index in hdf5 format
        violations (str): if specified, filename where to save violations in text
            format
        info (str): if specified, filename where to save run information in
            text format
        remove_after (bool): if set to True, will delete all the original 
            hms files after completion
    '''
    try:
        n_violated = 0
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
            if len(cviol) > 0:
                n_violated += 1
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
                    cviol = f['violations'][()]
                    _vprint(violations_file, cviol, i)
                    if len(cviol) > 0:
                        n_violated += 1

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

        return n_violated

    except: # let's make sure the hdf5 file is removed, else weird stuff happens
        if os.path.isfile(hss):
            os.remove(hss)
        raise
    







