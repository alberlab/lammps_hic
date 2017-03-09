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
The **lammps** module provides function to interface
with LAMMPS in order to perform the modeling
of the chromosome structures.

To perform a full modeling step, you may
want to check out the higher level wrappers
in the wrappers module.
'''


from __future__ import print_function, division
import os
import os.path
import h5py
import math
import logging
import numpy as np
import concurrent.futures
from numpy.linalg import norm
from io import StringIO
from subprocess import Popen, PIPE
try:
    import cPickle as pickle
except ImportError:
    import pickle


from .myio import read_hss, write_hss, read_full_actdist
from .util import monitor_progress, pretty_tdelta
from .globals import lammps_executable, float_epsilon


__author__  = "Guido Polles"
__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "polles@usc.edu"


ARG_DEFAULT = {
    'nucleus_radius': 5000.0,
    'occupancy': 0.2,
    'actdist': None,
    'fish': None,
    'damid': None,
    'out': 'out.lammpstrj',
    'data': 'input.data',
    'lmp': 'minimize.lam',
    'apprestr': None,
    'contact_kspring': 1.0,
    'contact_range': 2.0,
    'fish_type': 'rRpP',
    'fish_kspring': 1.0,
    'fish_tol': 0.0,
    'damid_kspring': 1.0,
    'damid_tol': 50.0,
    'mdsteps': 20000,
    'timestep': 0.25,
    'tstart': 20.0,
    'tstop': 1.0,
    'damp': 50.0,
    'seed': np.random.randint(100000000),
    'write': None,
    'thermo': 1000,
    'max_velocity': 5.0,
    'evfactor': 1.0,
    'max_neigh': 2000,
    'max_cg_iter': 500,
    'max_cg_eval': 500,
    'etol': 1e-4,
    'ftol': 1e-6,
    'territories': 0,
    'soft_min': 0,
    'ev_start': 0.0,
    'ev_stop': 0.0,
    'ev_step': 0,
}


class BondType(object):
    '''
    A bond type. The indexes are saved in
    the *0, ... , N-1* index, while its string
    representation is in the *1, ..., N* LAMMPS
    format.
    '''
    def __init__(self, b_id, type_str, kspring, r0):
        self.id = b_id
        self.kspring = kspring
        self.type_str = type_str
        self.r0 = r0 

    def __str__(self):
        return '{} {} {} {}'.format(self.id + 1,
                                    self.type_str,
                                    self.kspring,
                                    self.r0)

    def __eq__(self, other): 
        return (self.r0 == other.r0 and
                self.kspring == other.kspring and
                self.type_str == other.type_str)  


    def __hash__(self):
        return hash((self.r0, self.kspring, self.type_str))


class Bond(object):
    '''
    A bond between two atoms. The indexes are saved in
    the *0, ... , N-1* index, while its string
    representation is in the *1, ..., N* LAMMPS
    format.
    '''
    def __init__(self, b_id, bond_type, i, j):
        self.id = b_id
        self.bond_type = bond_type
        self.i = i
        self.j = j

    def __str__(self):
        return '{} {} {} {}'.format(self.id + 1,
                                    self.bond_type + 1,
                                    self.i + 1,
                                    self.j + 1)

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__


class BondContainer(object):
    '''
    A container to avoid keeping track of ids and duplicates.
    The dictionary for the types is to avoid re-use of 
    bond_types.
    
    This way, the check should be amortized O(1)
    '''
    def __init__(self):
        self.bond_types = []
        self.bonds = []
        self.bond_type_dict = {}

    def add_type(self, type_str, kspring=0.0, r0=0.0):
        '''
        Add a bond type.

        :Arguments:
            *type_str*
                Either 'harmonic_upper_bound' or 'harmonic_lower_bound'

            *kspring*
                Spring constant

            *r0*
                Activation distance (center to center)

        :Outputs:
            *bt*
                The corresponding BondType instance  
        '''
        bt = BondType(len(self.bond_types),
                      type_str,
                      kspring,
                      r0)

        old_id = self.bond_type_dict.get(bt)
        if old_id is None:
            self.bond_types.append(bt)
            self.bond_type_dict[bt] = bt.id
            return bt
        else: 
            return self.bond_types[old_id]

    def add_bond(self, bond_type, i, j):
        '''
        Add a bond of type *bond_type* between *i* and *j* beads

        :Arguments:
            *bond_type*
                A BondType instance returned by the add_type 
                function
            *i, j*
                Integer indexes of the two beads to be bound
        '''
        bond = Bond(len(self.bonds),
                    bond_type.id,
                    i,
                    j)
        self.bonds.append(bond)
        return bond


class DummyAtoms(object):
    '''
    The dummy atoms class is used to keep track of dummy atoms.

    They are usually connected to multiple atoms (to enforce, 
    for example, some distance from the cell center).
    If we use
    the same atom for too many bonds, the dummy atom will set the
    max_bonds parameter for all atoms to a very large number,
    letting memory usage explode.

    The solution is to have multiple dummy atoms and
    use a single atom only for a limited number of
    bonds.

    DummyAtoms.next() returns the same dummy atom id for 
    *DummyAtom.max_bonds* (default is 10) times, then updates
    the number of dummy atoms, making the new atom the 
    current one. 

    :Constructor Arguments:
        *n_atoms*
            The number of real atoms in the system.

    '''
    max_bonds = 10

    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.counter = 0
        self.n_dummy = 0

    def next(self):
        if self.counter % DummyAtoms.max_bonds == 0:
            self.n_dummy += 1
        self.counter += 1
        return self.n_dummy + self.n_atoms - 1


def _chromosome_string_to_numeric_id(chrom):
    '''
    Transform a list of strings in numeric
    ids (from 1 to N). Multiple chromosome copies
    will have the same id
    '''
    chr_map = {}
    chrom_id = []
    hv = 0
    for s in chrom:
        z = s.replace('chr', '')
        try:
            n = chr_map[z]
        except KeyError:
            n = hv + 1
            chr_map[z] = n
            hv = n
        chrom_id.append(n)
    return chrom_id


def generate_input(crd, bead_radii, chrom, **kwargs):
    '''
    This function generate two input files for
    the LAMMPS executable.

    :Arguments:
        *crd (numpy ndarray)*
            Starting bead coordinates

        *bead_radii (array or list of floats)*
            Radius of each bead in the system

        *chrom (array or list of strings)*
            Chromosome tag for each bead
        
    :Additional keyword arguments:
        *nucleus_radius (default=5000.0)*
            Radius of the nucleus in length units

        *occupancy (default=0.2)*
            Occupancy defined as the fraction of total volume 
            occupied by the beads

        *actdist (default=None)*
            Filename or array of activation distances.
            Activation distances are a list of records,
            where each record contains:

            - index of first bead

            - index of second bead

            - the desired contact probability

            - the activation distance

            - the corrected probability from the iterative correction

            - the present probability

        *fish (default=None)*
            Filename for a FISH target distances file

        *damid (default=None)*
            Filename or array of lamina DamID activation distances

        *out (default=out.lammpstrj)*
            Output filename for the lammps trajectory

        *data (default=input.data)*
            Output filename for the data file

        *lmp (default=minimize.lam)*
            Output filename for the LAMMPS script file

        *contact_kspring (default=1.0)*
            Spring costant for contact restraints

        *contact_range (default=2.0)*
            Distance in units of (:math:`r_{i} + r_{j}`) which defines a
            contact 

        *fish_type (default="rRpP")*
            if fish is set, determines the kind of restraints to 
            impose. If the string contains:

            - *r*: impose minimum radial positions

            - *R*: impose maximum radial positions

            - *p*: impose minimum pairwise distances

            - *P*: impose maximum pairwise distances

        *fish_kspring (default=1.0)*
            Spring constant for fish restraints

        *fish_tol (default=0.0)*
            Tolerance for fish restraints, in length units

        *damid_kspring (default=1.0)*
            Spring constant for DamID restraints

        *damid_tol (default=50.0)*
            Tolerance for DamID restraints

        *mdsteps (default=20000)*
            Number of molecular dynamics steps

        *timestep (default=0.25)*
            Timestep of Molecular Dynamics integrator

        *tstart (default=20.0)*
            Starting temperature (in :math:`k_BT`)

        *tstop (default=1.0)*
            Final temperature (in :math:`k_BT`)

        *damp (default=50.0)*
            Langevin damp parameter in timesteps

        *seed (default=np.random.randint(100000000))*
            Seed for the random number generator

        *write (default=None)*
            If set, dumps the coordinates on the trajectory file 
            every *write* timesteps

        *thermo (default=1000)*
            Print thermodynamic information to output
            every *thermo* timesteps

        *max_velocity (default=5.0)*
            Limit velocity on the NVT integrator (to avoid system explosions)

        *evfactor (default=1.0)*
            Scale the excluded volume potential by this factor

        *max_neigh (default=2000)*
            Set the maximum number of neighbor for each atom

        *max_cg_iter (default=500)*
            Maximum number of Conjugate Gradients iterations

        *max_cg_eval (default=500)*
            Maximum number of energy evaluations during 
            Conjugate Gradients

        *etol (default=1e-4)*
            Tolerance on energy to stop Conjugate Gradients

        *ftol (default=1e-6)*
            Tolerance on force to stop Conjugate Gradients

        *territories (default=0)*
            Deprecated.
            If larger than 0, select a bead every *territories* on each 
            chromosome, and constrain them to a short distance, to create
            chromosome territories

        *soft_min (default=0)*
            Deprecated.
            
            All simulations now use a soft potential. It can be tuned 
            setting *evfactor*

        *ev_step (default=0)*
            If an integer different from 0, the excluded volume interactions
            will be scaled linearly from *ev_start* to *ev_stop* during the
            simulation. The value of the scale parameter will be updated 
            every *ev_step* timesteps.

        *ev_start (default 0.0)*
            See *ev_step*

        *ev_stop (default 0.0)*
            See *ev_step*                     
    '''
    import numpy as np

    args = {k: v for k, v in ARG_DEFAULT.items()}
    for k, v in kwargs.items():
        if v is not None:
            args[k] = v

    if args['write'] is None:
        args['write'] = args['mdsteps']  # write only final step
    else:
        # this is only because it may be passed as string
        args['write'] = int(args['write'])

    chromid = _chromosome_string_to_numeric_id(chrom)

    n_hapl = len(crd) // 2
    n_atoms = len(crd)
    x1 = crd[:n_hapl]
    x2 = crd[n_hapl:]
    assert (len(x1) == len(x2))

    # Harmonic Upper Bond Coefficients are one for each bond type
    # and coded as:
    #   Bond_type_id kspring activation_distance
    # Each bond is coded as:
    #   Bond_id Bond_type_id ibead jbead

    bond_list = BondContainer()
    dummies = DummyAtoms(n_atoms)

    # first, determine atom types by radius
    atom_types_ids = {}
    at = 0
    atom_types = np.empty(n_atoms, dtype=int)
    for i, r in enumerate(bead_radii):
        aid = atom_types_ids.get(r)
        if aid == None:
            at += 1
            atom_types_ids[r] = at
            aid = at
        atom_types[i] = aid
    n_bead_types = len(atom_types_ids)

    ###############################
    # Add consecutive beads bonds #
    ###############################

    for i in range(n_atoms - 1):
        if chrom[i] == chrom[i + 1]:
            bt = bond_list.add_type('harmonic_upper_bound', 
                                    kspring=1.0, 
                                    r0=2*(bead_radii[i]+bead_radii[i+1]))
            bond_list.add_bond(bt, i, i + 1)

    ##############################
    # Add chromosome territories #
    ##############################

    if args['territories']:
        # for the territories step we want to make chromosomes very dense (3x?)
        terr_occupancy = args['occupancy'] * 3
        chrom_start = [0]
        for i in range(1, n_atoms):
            if chrom[i] != chrom[i - 1]:
                chrom_start.append(i)
        chrom_start.append(n_atoms)
        for k in range(len(chrom_start) - 1):
            nnbnds = 0
            chrom_len = chrom_start[k + 1] - chrom_start[k]
            chr_radii = bead_radii[chrom_start[k]:chrom_start[k + 1]]
            d = 4.0 * sum(chr_radii) / chrom_len  # 4 times the average radius
            d *= (float(chrom_len) / terr_occupancy)**(1. / 3)
            bt = bond_list.add_type('harmonic_upper_bound',
                                    kspring=1.0,
                                    r0=d)
            for i in range(chrom_start[k], chrom_start[k + 1], args['territories']):
                for j in range(i + args['territories'], chrom_start[k + 1], args['territories']):
                    bond_list.add_bond(bt, i, j)
                    nnbnds += 1

    ##############################################
    # Activation distance for contact restraints #
    ##############################################
    # NOTE: activation distances are surface to surface, not center to center
    # The old method was using minimum pairs
    # Here we enforce a single pair.
    if args['actdist'] is not None:
        if isinstance(args['actdist'], str) or isinstance(args['actdist'], unicode):
            if os.path.getsize(args['actdist']) > 0:
                actdists = read_full_actdist(args['actdist'])
            else:
                actdists = []
        else:
            actdists = args['actdist']

        for (i, j, pwish, d, p, pnow) in actdists:
            cd = np.zeros(4)
            rsph = bead_radii[i] + bead_radii[j]
            dcc = d + rsph  # center to center distance

            cd[0] = norm(x1[i] - x1[j])
            cd[1] = norm(x2[i] - x2[j])
            cd[2] = norm(x1[i] - x2[j])
            cd[3] = norm(x2[i] - x1[j])

            # find closest distance and remove incompatible contacts
            idx = np.argsort(cd)

            if cd[idx[0]] > dcc:
                continue

            # we have at least one bond to enforce
            bt = bond_list.add_type('harmonic_upper_bound',
                                    kspring=args['contact_kspring'],
                                    r0=args['contact_range'] * rsph)

            if idx[0] == 0 or idx[0] == 1:
                cd[2] = np.inf
                cd[3] = np.inf
            else:
                cd[0] = np.inf
                cd[1] = np.inf

            # add bonds
            if cd[0] < dcc:
                bond_list.add_bond(bt, i, j)
            if cd[1] < dcc:
                bond_list.add_bond(bt, i + n_hapl, j + n_hapl)
            if cd[2] < dcc:
                bond_list.add_bond(bt, i, j + n_hapl)
            if cd[3] < dcc:
                bond_list.add_bond(bt, i + n_hapl, j)

    #############################################
    # Activation distances for damid restraints #
    #############################################

    if args['damid'] is not None:
        if isinstance(args['damid'], str) or isinstance(args['damid'], unicode):
            if os.path.getsize(args['damid']) > 0:
                damid_actdists = np.genfromtxt(args['damid'],
                                               usecols=(0, 1, 2),
                                               dtype=(int, float, float))
                if damid_actdists.shape == ():
                    damid_actdists = [damid_actdists]
            else:
                damid_actdists = []

        for item in damid_actdists:
            i = item[0]
            d = item[2]
            # target minimum distance
            td = args['nucleus_radius'] - (2 * bead_radii[i])

            # current distances
            cd = np.zeros(2)
            cd[0] = args['nucleus_radius'] - norm(x1[i])
            cd[1] = args['nucleus_radius'] - norm(x2[i])

            idx = np.argsort(cd)

            if cd[idx[0]] > d:
                continue

            # we have at least one bond to enforce
            bt = bond_list.add_type('harmonic_lower_bound',
                                    kspring=args['damid_kspring'],
                                    r0=td)

            # add bonds
            if cd[0] < d:
                bond_list.add_bond(bt, i, dummies.next())
            if cd[1] < d:
                bond_list.add_bond(bt, i + n_hapl, dummies.next())

    ###############################################
    # Add bonds for fish restraints, if specified #
    ###############################################

    if args['fish'] is not None:
        hff = h5py.File(args['fish'], 'r')
        minradial = 'r' in args['fish_type']
        maxradial = 'R' in args['fish_type']
        minpair = 'p' in args['fish_type']
        maxpair = 'P' in args['fish_type']

        if minradial:
            cd = np.zeros(2)
            probes = hff['probes']
            minradial_dist = hff['radial_min'][:, args['i']]
            for k, i in enumerate(probes):
                td = minradial_dist[k]
                cd[0] = norm(x1[i])
                cd[1] = norm(x2[i])
                if cd[0] < cd[1]:
                    ii = i
                else:
                    ii = i + n_hapl
                bt = bond_list.add_type('harmonic_lower_bound',
                                        kspring=args['fish_kspring'],
                                        r0=max(0, td - args['fish_tol']))
                bond_list.add_bond(bt, ii, dummies.next())
                bt = bond_list.add_type('harmonic_upper_bound',
                                        kspring=args['fish_kspring'],
                                        r0=td + args['fish_tol'])
                bond_list.add_bond(bt, ii, dummies.next())

        if maxradial:
            cd = np.zeros(2)
            probes = hff['probes']
            maxradial_dist = hff['radial_max'][:, args['i']]
            for k, i in enumerate(probes):
                td = maxradial_dist[k]
                cd[0] = norm(x1[i])
                cd[1] = norm(x2[i])
                if cd[0] > cd[1]:
                    ii = i
                else:
                    ii = i + n_hapl
                bt = bond_list.add_type('harmonic_upper_bound',
                                        kspring=args['fish_kspring'],
                                        r0=td + args['fish_tol'])
                bond_list.add_bond(bt, ii, dummies.next())
                bt = bond_list.add_type('harmonic_lower_bound',
                                        kspring=args['fish_kspring'],
                                        r0=max(0, td - args['fish_tol']))
                bond_list.add_bond(bt, ii, dummies.next())

        if minpair:
            pairs = hff['pairs']
            minpair_dist = hff['pair_min'][:, args['i']]
            for k, (i, j) in enumerate(pairs):
                dmin = minpair_dist[k]

                cd = np.zeros(4)
                cd[0] = norm(x1[i] - x1[j])
                cd[1] = norm(x2[i] - x2[j])
                cd[2] = norm(x1[i] - x2[j])
                cd[3] = norm(x2[i] - x1[j])

                bt = bond_list.add_type('harmonic_lower_bound',
                                        kspring=args['fish_kspring'],
                                        r0=max(0, dmin - args['fish_tol']))
                bond_list.add_bond(bt, i, j)
                bond_list.add_bond(bt, i, j + n_hapl)
                bond_list.add_bond(bt, i + n_hapl, j)
                bond_list.add_bond(bt, i + n_hapl, j + n_hapl)

                bt = bond_list.add_type('harmonic_upper_bound',
                                        kspring=args['fish_kspring'],
                                        r0=dmin + args['fish_tol'])
                idx = np.argsort(cd)

                if idx[0] == 0 or idx[0] == 2:
                    ii = i
                else:
                    ii = i + n_hapl
                if idx[0] == 0 or idx[0] == 3:
                    jj = j
                else:
                    jj = j + n_hapl
                bond_list.add_bond(bt, ii, jj)

        if maxpair:
            pairs = hff['pairs']
            maxpair_dist = hff['pair_max'][:, args['i']]
            for k, (i, j) in enumerate(pairs):
                dmax = maxpair_dist[k]

                cd = np.zeros(4)
                cd[0] = norm(x1[i] - x1[j])
                cd[1] = norm(x2[i] - x2[j])
                cd[2] = norm(x1[i] - x2[j])
                cd[3] = norm(x2[i] - x1[j])

                bt = bond_list.add_type('harmonic_upper_bound',
                                        kspring=args['fish_kspring'],
                                        r0=dmax + args['fish_tol'])
                bond_list.add_bond(bt, i, j)
                bond_list.add_bond(bt, i, j + n_hapl)
                bond_list.add_bond(bt, i + n_hapl, j)
                bond_list.add_bond(bt, i + n_hapl, j + n_hapl)

                bt = bond_list.add_type('harmonic_lower_bound',
                                        kspring=args['fish_kspring'],
                                        r0=max(0, dmax - args['fish_tol']))
                idx = np.argsort(cd)
                if idx[-1] == 0 or idx[-1] == 2:
                    ii = i
                else:
                    ii = i + n_hapl
                if idx[-1] == 0 or idx[-1] == 3:
                    jj = j
                else:
                    jj = j + n_hapl
                bond_list.add_bond(bt, ii, jj)
        hff.close()

    ############################
    # Create LAMMPS input file #
    ############################

    n_bonds = len(bond_list.bonds)
    n_bondtypes = len(bond_list.bond_types)
    dummy_type = n_bead_types + 1

    with open(args['data'], 'w') as f:

        print('LAMMPS input\n', file=f)

        print(n_atoms + dummies.n_dummy, 'atoms\n', file=f)
        print(n_bead_types + 1, 'atom types\n', file=f)
        print(n_bondtypes, 'bond types\n', file=f)
        print(n_bonds, 'bonds\n', file=f)

        # keeping some free space to be sure
        print('-6000 6000 xlo xhi\n',
              '-6000 6000 ylo yhi\n',
              '-6000 6000 zlo zhi', file=f)

        print('\nAtoms\n', file=f)
        # index, molecule, atom type, x y z.
        for i, (x, y, z) in enumerate(crd):
            print(i + 1, chromid[i], atom_types[i], x, y, z, file=f)
        
        # dummy atoms in the middle
        dummy_mol = max(chromid) + 1
        for i in range(dummies.n_dummy):
            print(n_atoms + i + 1, dummy_mol, dummy_type, 0, 0, 0, file=f)

        print('\nBond Coeffs\n', file=f)
        for bc in bond_list.bond_types:
            print(bc, file=f)

        print('\nBonds\n', file=f)
        for b in bond_list.bonds:
            print(b, file=f)

        # Excluded volume coefficients
        at_radii = [x for x in sorted(atom_types_ids, key=atom_types_ids.__getitem__)]

        print('\nPairIJ Coeffs\n', file=f)
        for i, ri in enumerate(at_radii):
            for j in range(i, len(at_radii)):
                rj = at_radii[j] 
                dc = (ri + rj)
                A = (dc/math.pi)**2
                #sigma = dc / 1.1224 #(2**(1.0/6.0))
                #print(i+1, args['evfactor'], sigma, dc, file=f)
                print(i+1, j+1, A*args['evfactor'], dc, file=f)
        
        for i in range(len(at_radii) + 1):
            print(i+1, dummy_type, 0, 0, file=f)

    ##########################
    # Create lmp script file #
    ##########################

    with open(args['lmp'], 'w') as f:
        print('units                 lj', file=f)
        print('atom_style            bond', file=f)
        print('bond_style  hybrid',
              'harmonic_upper_bound',
              'harmonic_lower_bound', file=f)
        print('boundary              f f f', file=f)

        # Needed to avoid calculation of 3 neighs and 4 neighs
        print('special_bonds lj/coul 1.0 1.0 1.0', file=f)


        # excluded volume
        print('pair_style soft', 2.0 * max(bead_radii), file=f)  # global cutoff

        print('read_data', args['data'], file=f)
        print('mass * 1.0', file=f)

        print('group beads type !=', dummy_type, file=f)
        print('group dummy type', dummy_type, file=f)

        print('neighbor', max(bead_radii), 'bin', file=f)  # skin size
        print('neigh_modify every 1 check yes', file=f)
        print('neigh_modify one', args['max_neigh'],
              'page', 20 * args['max_neigh'], file=f)

        # Freeze dummy atom

        print('neigh_modify exclude group dummy all', file=f)
        print('fix 1 dummy setforce 0.0 0.0 0.0', file=f)

        # Integration
        # select the integrator
        print('fix 2 beads nve/limit', args['max_velocity'], file=f)
        # Impose a thermostat - Tstart Tstop tau_decorr seed
        print('fix 3 beads langevin', args['tstart'], args['tstop'],
              args['damp'], args['seed'], file=f)
        print('timestep', args['timestep'], file=f)

        # ramp excluded volume
        if args['ev_step'] != 0:
            print('variable evprefactor equal '
                  'ramp(%f,%f)' % (args['ev_start'], args['ev_stop']),
                  file=f)
            print('fix 4 all adapt',
                  args['ev_step'],
                  'pair soft a * * v_evprefactor scale yes', file=f)

        # Region
        print('region mySphere sphere 0.0 0.0 0.0',
              args['nucleus_radius'] + 2 * max(bead_radii), file=f)
        print('fix wall beads wall/region mySphere harmonic 10.0 1.0 ',
              2 * max(bead_radii), file=f)

        #print('pair_modify shift yes mix arithmetic', file=f)

        # outputs:
        print('dump   id beads custom',
              args['write'],
              args['out'],
              'id type x y z fx fy fz', file=f)

        # Thermodynamic info style for output
        print('thermo_style custom step temp epair ebond', file=f)
        print('thermo', args['thermo'], file=f)

        # Run MD
        print('run', args['mdsteps'], file=f)

        # Run CG
        print('min_style cg', file=f)
        print('minimize', args['etol'], args['ftol'],
              args['max_cg_iter'], args['max_cg_eval'], file=f)

    return bond_list


def _check_violations(bond_list, crd):
    violations = []
    for bond in bond_list.bonds:
        bt = bond_list.bond_types[bond.bond_type]
        d = norm(crd[bond.i] - crd[bond.j])
        if bt.type_str == 'harmonic_upper_bound':
            if d > bt.r0:
                absv = d - bt.r0
                relv = (d - bt.r0) / bt.r0
                if relv > 0.05:
                    violations.append((bond.i, bond.j, absv, relv, str(bt)))
        if bt.type_str == 'harmonic_lower_bound':
            if d < bt.r0:
                absv = bt.r0 - d
                relv = (bt.r0 - d) / bt.r0
                if relv > 0.05:
                    violations.append((bond.i, bond.j, absv, relv, str(bt)))
    return violations


def _reverse_readline(fh, buf_size=8192):
    """a generator that returns the lines of a file in reverse order"""
    segment = None
    offset = 0
    fh.seek(0, os.SEEK_END)
    file_size = remaining_size = fh.tell()
    while remaining_size > 0:
        offset = min(file_size, offset + buf_size)
        fh.seek(file_size - offset)
        buffer = fh.read(min(remaining_size, buf_size))
        remaining_size -= buf_size
        lines = buffer.split('\n')
        # the first line of the buffer is probably not a complete line so
        # we'll save it and append it to the last line of the next buffer
        # we read
        if segment is not None:
            # if the previous chunk starts right from the beginning of line
            # do not concact the segment to the last line of new chunk
            # instead, yield the segment first
            if buffer[-1] is not '\n':
                lines[-1] += segment
            else:
                yield segment
        segment = lines[0]
        for index in range(len(lines) - 1, 0, -1):
            if len(lines[index]):
                yield lines[index]
    # Don't yield None if the file was empty
    if segment is not None:
        yield segment


def get_info_from_log(output):
    ''' gets final energy, excluded volume energy and bond energy.
    TODO: get more info? '''
    info = {}
    generator = _reverse_readline(output)

    for l in generator:
        if l[:9] == '  Force t':
            ll = next(generator)
            info['final-energy'] = float(ll.split()[2])
            break

    for l in generator:
        if l[:4] == 'Loop':
            ll = next(generator)
            _, _, epair, ebond = [float(s) for s in ll.split()]
            info['pair-energy'] = epair
            info['bond-energy'] = ebond
            break
    
    # EN=`grep -A 1 "Energy initial, next-to-last, final =" $LAMMPSLOGTMP \
    # | tail -1 | awk '{print $3}'`
    return info


def get_last_frame(fh):
    atomlines = []
    for l in _reverse_readline(fh):
        if 'ITEM: ATOMS' in l:
            v = l.split()
            ii = v.index('id') - 2
            ix = v.index('x') - 2
            iy = v.index('y') - 2
            iz = v.index('z') - 2
            break
        atomlines.append(l)

    crds = np.empty((len(atomlines), 3))
    for l in atomlines:
        v = l.split()
        i = int(v[ii]) - 1  # ids are in range 1-N
        x = float(v[ix])
        y = float(v[iy])
        z = float(v[iz])
        crds[i][0] = x
        crds[i][1] = y
        crds[i][2] = z

    return crds


def lammps_minimize(crd, radii, chrom, run_name, tmp_files_dir='/dev/shm', log_dir='.', check_violations=True, **kwargs):
    '''
    Actual minimization wrapper.
    
    After creating the input and data files for lammps,
    runs the lammps executable in a process (using subprocess.Popen).

    This function is internally called by the higher level 
    parallel wrappers.

    When the program returns, it parses the output and returns the 
    results.

    :Arguments:
        *crd (numpy array)*
            Initial coordinates (n_beads x 3 array).

        *radii (list of floats)*
            Radius for each particle in the system

        *chrom (list of strings)*
            Chromosome name for each particle. Note that chain
            breaks are determined by changes in the chrom
            value

        *run_name (string)* 
            Name of the run - determines only the name of temporary 
            files.

        *tmp_files_dir (string)*
            Location of temporary files. Of course, it needs to be 
            writable. 

            The default value, **/dev/shm** is usually a in-memory file 
            system, useful to share data between processes without
            actually writing to a physical disk.

            The function checks for exceptions and try to remove
            the temporary files. If the interpreter is killed without
            being able to catch exceptions (for example because of
            a walltime limit) some files could be left behind.

        *log_dir (string)*
            Deprecated, no effect at all.

        *check_violations (bool)*
            Perform a check on the violations on the assigned 
            bonds. If set to **False**, the check is skipped
            and the violation output will be an empty list.

        *\*\*kwargs*
            Optional minimization arguments. (see lammps.generate_input)

    :Output:
        *new_crd (numpy ndarray)*
            Coordinates after minimization.

        *info (dict)*
            Dictionary with summarized info for the run, as returned by
            lammps.get_info_from_log.

        *violations (list)*
            List of violations. If the check_violations parameter is set
            to False, returns an empty list.

    Exceptions:
        If the lammps executable return code is different from 0, it
        raises a RuntimeError with the contents of the standard error.
    '''

    data_fname = os.path.join(tmp_files_dir, run_name + '.data')
    script_fname = os.path.join(tmp_files_dir, run_name + '.lam')
    traj_fname = os.path.join(tmp_files_dir, run_name + '.lammpstrj')

    try:

        # prepare input
        opts = {'out': traj_fname, 'data': data_fname, 'lmp': script_fname}
        kwargs.update(opts)
        bond_list = generate_input(crd, radii, chrom, **kwargs)

        # run the lammps minimization
        with open(script_fname, 'r') as lamfile:
            proc = Popen([lammps_executable, '-log', '/dev/null'],
                         stdin=lamfile,
                         stdout=PIPE,
                         stderr=PIPE)
            output, error = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError('LAMMPS exited with non-zero exit code: %d\nStandard Error:\n%s\n', 
                               proc.returncode, 
                               error)

        # get results
        info = get_info_from_log(StringIO(unicode(output)))
        with open(traj_fname, 'r') as fd:
            new_crd = get_last_frame(fd)

        if os.path.isfile(data_fname):
            os.remove(data_fname)
        if os.path.isfile(script_fname):
            os.remove(script_fname)
        if os.path.isfile(traj_fname):
            os.remove(traj_fname)

    except:
        if os.path.isfile(data_fname):
            os.remove(data_fname)
        if os.path.isfile(script_fname):
            os.remove(script_fname)
        if os.path.isfile(traj_fname):
            os.remove(traj_fname)
        raise


    if check_violations:
        if info['bond-energy'] > float_epsilon:
            violations = _check_violations(bond_list, new_crd)
        else: 
            violations = []
        return new_crd, info, violations
    else:
        return new_crd, info, []


# closure for parallel mapping
def parallel_fun(radius, chrom, tmp_files_dir='/dev/shm', log_dir='.', check_violations=True, **kwargs):
    def inner(pargs):
        try:
            crd, run_name = pargs
            return lammps_minimize(crd,
                                   radius,
                                   chrom,
                                   run_name,
                                   tmp_files_dir=tmp_files_dir,
                                   log_dir=log_dir,
                                   check_violations=check_violations,
                                   **kwargs)
        except:
            import sys
            return (None, str(sys.exc_info()))
    return inner


def bulk_minimize(parallel_client,
                  crd_fname,
                  prefix='minimize',
                  tmp_files_dir='/dev/shm',
                  log_dir='.',
                  check_violations=True,
                  ignore_restart=False,
                  **kwargs):    
    try:
        logger = logging.getLogger()

        logger.debug('bulk_minimize() entry')

        engine_ids = list(parallel_client.ids)
        parallel_client[:].use_cloudpickle()

        logger.debug('bulk_minimize(): reading data')
        crd, radii, chrom, n_struct, n_bead = read_hss(crd_fname)
        logger.debug('bulk_minimize(): done reading data')

        lbv = parallel_client.load_balanced_view(engine_ids)

        if ignore_restart:
            to_minimize = list(range(n_struct))
            completed = []
        else:
            if os.path.isfile(prefix + '.incomplete.pickle'):
                logger.info('bulk_minimize(): Found restart file.')
                with open(prefix + '.incomplete.pickle', 'rb') as pf:
                    completed, errors = pickle.load(pf)
                    to_minimize = [i for i, e in errors]
                logger.info('bulk_minimize(): %d structures yet to minimize', len(to_minimize))
            else:
                to_minimize = list(range(n_struct))
                completed = []

        logger.debug('bulk_minimize(): preparing arguments and function')
        pargs = [(crd[i], prefix + '.' + str(i)) for i in to_minimize] 
        f = parallel_fun(radii, chrom, tmp_files_dir, log_dir, check_violations=check_violations, **kwargs)
        
        logger.info('bulk_minimize(): Starting bulk minimization of %d structures on %d workers', len(to_minimize), len(engine_ids))

        ar = lbv.map_async(f, pargs)

        logger.debug('bulk_minimize(): map_async sent.')

        monitor_progress('bulk_minimize() - %s' % prefix, ar)

        logger.info('bulk_minimize(): Done')

        results = list(ar.get())

        ar = None

        errors = [ (i, r[1]) for i, r in zip(to_minimize, results) if r[0] is None]
        completed += [ (i, r) for i, r in zip(to_minimize, results) if r[0] is not None]
        completed = list(sorted(completed))

        if len(errors):
            for i, e in errors:
                logger.error('Exception returned in minimizing structure %d: %s', i, e)
            logger.info('Saving partial results to %s.incomplete.pickle')
            with open(prefix + '.incomplete.pickle', 'wb') as pf:
                pickle.dump((completed, errors), pf, pickle.HIGHEST_PROTOCOL)
            raise RuntimeError('Unable to complete bulk_minimize()')

        # We finished lammps runs here
        new_crd = np.array([x for i, (x, _, _) in completed])

        if os.path.isfile(prefix + '.incomplete.pickle'):
            os.remove(prefix + '.incomplete.pickle')

        energies = np.array([info['final-energy'] for i, (_, info, _) in completed])
        violated = [(i, info, violations) for i, (_, info, violations) in completed if len(violations)]
        n_violated = len(violated)
        logger.info('%d structures with violations over %d structures',
                     n_violated,
                     len(completed))
        
        write_hss(prefix + '.hss', new_crd, radii, chrom)
        np.savetxt(prefix + '_energies.dat', energies)
        return completed
    
    except:
        logger.error('bulk_minimization() failed', exc_info=True)
        raise

    
def _serial_lammps_call(largs):
    '''
    Serial function to be mapped in parallel.

    It is intended to be used only internally by parallel routines.

    :Arguments: 
        *largs (tuple)* 
            Triplet of filenames: (from, parameters, to). *from* and *to*
            are hms files, parameters is a json dictionary of arguments
            to the lammps_minimize call
    
    :Output:
        *returncode*
            0 if success, None otherwise

        *info*
            Dictionary of info returned by lammps_minimize. If the 
            run fails, info is set to the formatted traceback
            string

        *n_violations*
            Number of violated restraint at the end of minimization;
            is set to zero in case of failure.

    :Exceptions:
        In case of failure, gracefully returns None and the traceback
        (see above)
    '''
    try:
        # importing here so it will be called on the parallel workers
        import json
        import traceback
        import os.path
        from .myio import read_hms, write_hms
        
        # unpack arguments
        fname, param_fname, new_fname = largs
        run_name = os.path.splitext(os.path.basename(new_fname))[0]

        # read parameters from disk
        with open(param_fname) as f:
            kwargs = json.load(f)


        # read coordinates from disk
        crd, radii, chrom, n_bead, violations, info = read_hms(fname)
    
        # perform minimization
        new_crd, info, violations = lammps_minimize(crd, radii, chrom, run_name, **kwargs)
        
        # write coordinates to disk asyncronously and return run info
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        ex.submit(write_hms, new_fname, new_crd, radii, chrom, violations, info)
        return (0, info, len(violations))

    except:
        import sys
        etype, value, tb = sys.exc_info()
        infostr = ''.join(traceback.format_exception(etype, value, tb))
        return (None, infostr, 0)

    

def bulk_minimize_single_file(parallel_client,
                              old_prefix,
                              new_prefix,
                              n_struct,
                              workdir='.',
                              tmp_files_dir='/dev/shm',
                              log_dir='.',
                              check_violations=True,
                              ignore_restart=False,
                              **kwargs):
    '''
    Uses ipyparallel to minimize a population of structures.

    :Arguments:
        *parallel_client (ipyparallel.Client instance)* 
            The ipyparallel.Client instance to send the jobs to

        *old_prefix (string)*
            The function will search for files named 
            *old_prefix_<n>.hms* in the working directory, with *n*
            in the range 0, ..., *n_struct*

        *new prefix (string)*
            After minimization, *n_struct* new files 
            named *new_prefix_<n>.hms* will be written to the
            working directory

        *n_struct (int)*
            Number of structures in the population

        *workdir (string)*
            Set the working directory where the hms files will
            be found and written.

        *tmp_files_dir (string)*
            Directory where to store temporary files

        *log_dir (string)*
            Eventual errors will be dumped to this directory

        *check_violations (bool)*
            If set to False, skip the violations check

        *ignore_restart (bool)*
            If set to True, the function will not check the
            presence of files named *new_prefix_<n>.hms*
            in the working directory. All the runs will be 
            re-sent and eventual files already present 
            will be overwritten.

        *\*\*kwargs*
            Other keyword arguments to pass to the minimization
            process. See lammps.generate_input documentation for
            details

    :Output:
        *now_completed*
            Number of completed minimizations in this call

        *n_violated*
            Number of structures with violations in this call

        Additionally, *now_completed* files 
        named *new_prefix_<n>.hms*
        will be written in the *work_dir* directory.
    
    :Exceptions:
        If one or more of the minimizations
        fails, it will raise a RuntimeError.
    '''

    # get the logger
    logger = logging.getLogger()

    try:
        import json

        # write all parameters to a file on filesystem
        kwargs['check_violations'] = check_violations
        kwargs['ignore_restart'] = ignore_restart
        kwargs['tmp_files_dir'] = tmp_files_dir
        kwargs['log_dir'] = log_dir
        parameter_fname = os.path.join(workdir, new_prefix + '.parms.json')
        with open(parameter_fname, 'w') as fp:
            json.dump(kwargs, fp)
    
        # check if we already have some output files written.
        completed = []
        to_minimize = []
        if ignore_restart:
            to_minimize = list(range(n_struct))
            logger.info('bulk_minimize(): ignoring completed runs.')
        else:
            for i in range(n_struct):
                fname = '{}_{}.hms'.format(new_prefix, i)
                fpath = os.path.join(workdir, fname)
                if os.path.isfile(fpath):
                    completed.append(i)
                else:
                    to_minimize.append(i)
            if len(to_minimize) == 0:
                logger.info('bulk_minimize(): minimization appears to be already finished.')
                return (0, 0)
            logger.info('bulk_minimize(): Found %d already minimized structures. Minimizing %d remaining ones', len(completed), len(to_minimize))

        # prepare arguments as a list of tuples (map wants a single argument for each call)
        pargs = [(os.path.join(workdir, '{}_{}.hms'.format(old_prefix, i)),
                  parameter_fname,
                  os.path.join(workdir, '{}_{}.hms'.format(new_prefix, i))) for i in to_minimize] 

        # using ipyparallel to map functions to workers
        engine_ids = list(parallel_client.ids)
        lbv = parallel_client.load_balanced_view()

        logger.info('bulk_minimize(): Starting bulk minimization of %d structures on %d workers', len(to_minimize), len(engine_ids))
        ar = lbv.map_async(_serial_lammps_call, pargs)
        logger.debug('bulk_minimize(): map_async sent.')

        # monitor progress
        monitor_progress('bulk_minimize() - %s' % new_prefix, ar)

        # post-analysis
        logger.info('bulk_minimize(): Done (Total time: %s)' % pretty_tdelta(ar.wall_time))
        results = list(ar.get())  # returncode, info, n_violations  

        # check for errors
        errors = [ (i, r[1]) for i, r in zip(to_minimize, results) if r[0] is None]
        now_completed = [ (i, r) for i, r in zip(to_minimize, results) if r[0] is not None]
        if len(errors):
            for i, e in errors:
                logger.error('Exception returned in minimizing structure %d: %s', i, e)
                print()
            raise RuntimeError('Unable to complete bulk_minimize()')
        
        # print a summary
        n_violated = sum([1 for i, r in now_completed if r[2] > 0 ])
        logger.info('%d structures with violations over %d structures',
                     n_violated,
                     len(now_completed))

        return len(now_completed), n_violated
    
    except:
        logger.error('bulk_minimization() failed', exc_info=True)
        raise
