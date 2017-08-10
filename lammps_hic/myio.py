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
from alabtools.utils import Index, Genome, HssFile, COORD_DTYPE, RADII_DTYPE


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


def read_hms(fname, group):
    '''
    Reads an hms file.
    Arguments:
        fname (str): hms file path

    Returns:
        crd (np.ndarray(dtype='f4')): coordinates of the structure
        radii (np.ndarray(dtype='f4')): radius for each bead
        chrom (np.ndarray(dtype='int32')): chromosome tag for each bead
        n_bead (int): the number of beads in the structure
        violations (np.recarray): an array containing the violations.
            Refer to the lammps.check_violations function documentation.
        info (dict): a dictionary containing run informations.
    '''
    
    with h5py.File(fname, 'r') as ff:
        radii = ff['radii'][()]
        index = ff['index'][()]
        n_bead = ff['nbead'][()]
        g = ff[group]
        crd = g['coordinates'][()]
        violations = g['violations'][()]
        info = json.loads(g['info'][()])
    return crd, radii, index, n_bead, violations, info


def write_hms(fname, group, crd, radii, chrom, violations=[], info={}, overwrite=False):
    '''
    TODO: write docs
    '''
    if len(radii) != len(crd):
        logging.warning('write_hms(): len(radii) != crd.shape[0]')

    if len(chrom) != len(crd):
        logging.warning('write_hms(): len(chrom) != crd.shape[0]')
    
    with h5py.File(fname, 'r+') as ff:
        if group in ff:
            if overwrite:
                del ff['group']
            else:
                raise RuntimeError('group %s already present' % group)
        f = ff.create_group(group)
        f.create_dataset('coordinates', data=crd, dtype='f4', compression='gzip')
        f.create_dataset('radius', data=radii, dtype='f4')
        f.create_dataset('idx', data=chrom)
        f.create_dataset('nbead', data=crd.shape[0], dtype='i4')
        f.create_dataset('violations', data=np.array(violations, dtype=violations_dtype), compression='gzip')
        f.create_dataset('info', data=json.dumps(info), compression='gzip')


def remove_hms(prefix, n_struct):
    '''
    Removes hms groups.
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


def pack_hms(fname, prefix, n_struct, hss=None, violations=None, info=None, remove_after=False):
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
        crd, radii, chrom, n_beads, cviol, cinfo = read_hms('{}_{}.hms'.format(fname, 0))
        
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
    

class Violations(object):
    def __init__(self, obj=None):
        if obj is None:
            self.ii = np.array([], dtype='int32')
            self.jj = np.array([], dtype='int32')
            self.abs_vals = np.array([], dtype='float32')
            self.rel_vals = np.array([], dtype='float32')
            self.bond_types = np.array([], dtype='int32')
        elif isinstance(obj, h5py.Group):
            self.ii = obj['ii'][:]
            self.jj = obj['jj'][:]
            self.abs_vals = obj['abs_vals'][:]
            self.rel_vals = obj['rel_vals'][:]
            self.bond_types = obj['bond_types'][:]
        else:
            raise ValueError('Wrong object type in Violations constructor')

    def add(self, i, j, abs_val, rel_val, bond_type):
        np.append(self.ii, i) 
        np.append(self.jj, j) 
        np.append(self.abs_vals, abs_val) 
        np.append(self.rel_vals, rel_val) 
        np.append(self.bond_types, bond_type)

    def save(self, grp):
        assert isinstance(grp, h5py.Group)
        g = grp.create_group('violations')
        g.create_dataset('ii', data=self.ii)
        g.create_dataset('jj', data=self.jj)
        g.create_dataset('bondtypes', data=self.bond_types)
        g.create_dataset('abs_vals', data=self.abs_vals)
        g.create_dataset('rel_vals', data=self.rel_vals)

    def __len__(self):
        return len(self.ii)


class HtfFile(h5py.File):
    
    '''
    Hdf5 Temporary File
    Similar to an .hms file from Nan, but with more resource economy.
    Radii and indexes are saved once, no pdb structures are saved.
    '''
    
    def init(self, radii, index):
        self.create_dataset('radii', data=radii, dtype=RADII_DTYPE)
        self.attrs.create('nbead', len(radii), dtype='int32')
        index.save(self)

    def get_index(self):
        return Index(self)

    def set_index(self, index):
        assert isinstance(index, Index)
        index.save(self)

    def get_radii(self):
        return self['radii'][:]

    def get_coordinates(self, group):
        return self[group]['coordinates'][:]

    def set_coordinates(self, group, crd):
        if group not in self:
            g = self.create_group(group)
        else:
            g = self[group]
        if 'coordinates' not in g:
            g.create_dataset('coordinates', data=crd, 
                             dtype=COORD_DTYPE)
        else:
            g['cooordinates'][...] = crd

    def get_violations(self, group):
        return Violations(self[group])

    def set_violations(self, group, violations):
        if group not in self:
            g = self.create_group(group)
        else:
            g = self[group]
        violations.save(g)

    def get_info(self, group):
        return json.loads(self[group]['info'][:])

    def set_info(self, group, info):
        if group not in self:
            g = self.create_group(group)
        else:
            g = self[group]
        if 'info' not in g:
            g.create_dataset('info', data=json.dumps(info))
        else:
            g['info'][...] = json.dumps(info)

    def set_radii(self, radii):
        assert isinstance(radii, np.ndarray)
        if len(radii.shape) != 1:
            raise ValueError('radii should be a one dimensional array')
        
        if 'radii' in self:
            self['radii'][...] = radii
        else:
            self.create_dataset('radii', data=radii, dtype=RADII_DTYPE)
            self.attrs.create('nbead', len(radii), dtype='int32')

    def get_nbead(self):
        return self.attrs['nbead']

    radii = property(get_radii, set_radii)
    index = property(get_index, set_index, doc='a alabtools.Index instance')
    nbead = property(get_nbead)




class PopulationCrdFile(object):

    '''
    A file which contains coordinates for a population. 
    It performs lazy evaluations of "columns" (structure coordinates) and 
    "rows" (coordinates of a single bead across the population).
    Use only rows or only columns! This is not yet smart enough to update
    the other in-memory coordinates. Close and reopen to change from
    rows to columns and vice versa.
    The max_memory parameter controls the maximum amount of memory the object
    should use. Note that the memory is checked only on the raw data,
    does not take into consideration overhead for python structures and 
    internal variables.
    
    Parameters
    ----------
        fname : string
            Path of the file on disk
        mode : string
            Either 'r' (readonly), 'w' (create/truncate), or 'r+' (read
            and write)
        shape : list
            Dimensions of the population coordinates (n_struct x n_bead x 3).
            Ignored if in readonly/rw modes.
        dtype : numpy.dtype or string
            Data type. Defaults to 'float32'
        max_memory : int
            Maximum number of bytes to use in memory. Note that does not 
            take into consideration data structures overheads and similar.
            Will use slightly more memory than this maximum.
    '''

    def __init__(self, fname, mode='r+', shape=(0, 0, 3), dtype='float32', max_memory=int(1e9)):
        
        assert mode == 'r+' or mode == 'w' or mode =='r'

        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.datasize = self.dtype.itemsize * np.prod(shape)
        
        self.fd = open(fname, mode + 'b')
        if (mode == 'w'):
            assert len(shape) == 3
            self.shape = shape
            self.headersize = 4 + 4*(3)
            if (self.datasize > 0):
                self.fd.seek(self.datasize+self.headersize-1)
                self.fd.write(b"\0")
            self._write_header()
        else:
            ndim = np.fromfile(self.fd, dtype=np.int32, count=1)
            assert ndim == 3
            self.shape = np.fromfile(self.fd, dtype=np.int32, count=3)
            self.headersize = 4 + 4*(3)

        self.nstruct = self.shape[0]
        self.nbead = self.shape[1]

        self.crd3size = self.itemsize * 3
        self.structsize = self.crd3size * self.nbead
        self.beadsize =  self.crd3size * self.nstruct


        self.beads = {}
        self.structs = {}

        self.cq = []
        self.bq = []
        self._cwrt = {}
        self._bwrt = {}
        self.max_memory = max_memory

        assert self.structsize <= max_memory and self.beadsize <= max_memory

    def __enter__(self):
        return self

    def used_memory(self):
        return (self.structsize * len(self.structs) + 
                self.beadsize * len(self.beads))

    def dump_coord(self, idx):
            self.fd.seek(self.headersize + self.structsize * idx)
            self.structs[idx].tofile(self.fd)

    def free_coord(self, idx):
        if idx in self._cwrt:
            self.dump_coord(idx)
            del self._cwrt[idx]
        del self.structs[idx]

    def dump_bead(self, idx):
        v = self.beads[idx]
        for i in range(self.nstruct):
            self.fd.seek(self.headersize + i * self.structsize + 
                         idx * self.crd3size)
            v[i].tofile(self.fd)

    def free_bead(self, idx):
        if idx in self._cwrt:
            self.dump_bead(idx)
            del self._bwrt[idx]       
        del self.beads[idx]

    def free_space(self, requested):
        assert requested <= self.max_memory
        while self.used_memory() + requested > self.max_memory:
            if len(self.structs):
                ir = self.cq.pop()
                self.free_coord(ir)
            else:
                ir = self.bq.pop()
                self.free_bead(ir)

    def get_struct(self, idx):
        idx = int(idx)
        if idx in self.structs:
            pos = self.cq.index(idx)
            self.cq.insert(0, self.cq.pop(pos))
            return self.structs[idx]
        else:
            self.free_space(self.structsize)
            self.fd.seek(self.headersize + self.structsize * idx)
            self.structs[idx] = np.fromfile(self.fd, count=self.nbead * 3, 
                                          dtype=self.dtype).reshape(
                                            (self.nbead, 3))
            self.cq.insert(0, idx)
            return self.structs[idx] 

    def read_bead(self, idx):
        v = self.beads[idx]
        for i in range(self.nstruct):
            self.fd.seek(self.headersize + self.structsize * i + 
                        self.crd3size * idx)
            v[i] = np.fromfile(self.fd, count=3, dtype=self.dtype)

    def get_bead(self, idx):
        idx = int(idx)
        if idx in self.beads:
            pos = self.bq.index(idx)
            self.bq.insert(0, self.bq.pop(pos))
            return self.beads[idx]
        else:
            self.free_space(self.beadsize)
            self.beads[idx] = np.empty((self.nstruct, 3))
            self.read_bead(idx)
            self.bq.insert(0, idx)
            return self.beads[idx]

    def _write_header(self):
        self.fd.seek(0)
        np.int32(len(self.shape)).tofile(self.fd)
        for dim in self.shape:
            np.int32(dim).tofile(self.fd)

    def set_struct(self, idx, struct):
        idx = int(idx)
        assert struct.shape == (self.nbead, 3)
        if idx in self.structs:
            pos = self.cq.index(idx)
            self.cq.insert(0, self.cq.pop(pos))    
        else:
            self.free_space(self.structsize)
            self.cq.insert(0, idx)
        
        self._cwrt[idx] = True
        self.structs[idx] = struct.astype(self.dtype, 'C', 'same_kind')

    def set_bead(self, idx, bead):
        idx = int(idx)
        assert bead.shape == (self.nstruct, 3)
        if idx in self.beads:
            pos = self.bq.index(idx)
            self.bq.insert(0, self.bq.pop(pos))    
        else:
            self.free_space(self.beadsize)
            self.bq.insert(0, idx)
        
        self._bwrt[idx] = True
        self.beads[idx] = bead.astype(self.dtype, 'C', 'same_kind')

    def get_all(self): # ignores limits
        self.flush()
        self.fd.seek(self.headersize)
        return np.fromfile(self.fd, count=self.nbead * self.nstruct * 3, 
                           dtype=self.dtype).reshape(
                           self.shape)

    def flush(self):
        for c in self._cwrt:
            self.dump_coord(c)
        for b in self._bwrt:
            self.dump_bead(b)

    def close(self):
        try:
            self.fd.close()
        except:
            pass

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            self.flush()
            self.fd.close()
        except:
            pass
