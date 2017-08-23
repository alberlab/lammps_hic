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
import math
import logging
import numpy as np
from numpy.linalg import norm
from io import StringIO
from myio import Violations
from itertools import groupby

from subprocess import Popen, PIPE
try:
    import cPickle as pickle
except ImportError:
    import pickle


from .myio import read_hss, write_hss, HtfFile
from .util import monitor_progress, pretty_tdelta
from .globals import lammps_executable, float_epsilon
from .restraints import *
from .lammps_utils import *


__author__  = "Guido Polles"
__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "polles@usc.edu"


ARG_DEFAULT = [
    ('nucleus_radius', 5000.0, float, 'default nucleus radius'),
    ('occupancy', 0.2, float, 'default volume occupancy (from 0.0 to 1.0)'),
    ('actdist', None, str, 'activation distances file (ascii text)'),
    ('fish', None, str, 'fish distances file (hdf5)'),
    ('damid', None, str, 'damid activation distances file (ascii text)'),
    ('bc_cluster', None, str, 'filename of cluster file (hdf5)'),
    ('bc_cluster_size', 5.0, float, 'inverse of volume occupancy'),
    ('out', 'out.lammpstrj', str, 'Temporary lammps trajectory file name'),
    ('data', 'input.data', str, 'Temporary lammmps input data file name'), 
    ('lmp', 'minimize.lam', str, 'Temporary lammps script file name'), 
    ('apprestr', None, str, 'Deprecated'),
    ('contact_kspring', 1.0, float, 'HiC contacts spring constant'),
    ('contact_range', 2.0, float, 'HiC contact range (in radius units)'),
    ('fish_type', 'rRpP', str, 'FISH restraints type'),
    ('fish_kspring', 1.0, float, 'FISH restraints spring constant'),
    ('fish_tol', 0.0, float, 'FISH restraints tolerance (nm)'),
    ('damid_kspring', 1.0, float, 'lamina DamID restr. spring constant'),
    ('damid_tol', 50.0, float, 'lamina DamID restraint tolerance (nm)'),
    ('mdsteps', 20000, int, 'Number of MD steps per round'),
    ('timestep', 0.25, float, 'MD timestep'),
    ('tstart', 20.0, float, 'MD initial temperature'),
    ('tstop', 1.0, float, 'MD final temperature'),
    ('damp', 50.0, float, 'MD damp parameter'),
    ('seed', np.random.randint(100000000), int, 'RNG seed'),
    ('write', -1, int, 'Dump coordinates every <write> MD timesteps'),
    ('thermo', 1000, int, 'Output thermodynamic info every <thermo>' 
                          ' MD timesteps'),
    ('max_velocity', 5.0, float, 'Cap particle velocity'),
    ('evfactor', 1.0, float, 'Scale excluded volume by this factor'),
    ('max_neigh', 2000, int, 'Maximum numbers of neighbors per particle'),
    ('max_cg_iter', 500, int, 'Maximum # of Conjugate Gradient steps'),
    ('max_cg_eval', 500, int, 'Maximum # of Conjugate Gradient evaluations'),
    ('etol', 1e-4, float, 'Conjugate Gradient energy tolerance'),
    ('ftol', 1e-6, float, 'Conjugate Gradient force tolerance'),
    ('territories', 0, int, 'apply territories restraints every <> beads'),
    ('soft_min', 0, int, 'perform a soft minimization of lenght <> timesteps'),
    ('ev_start', 0.0, float, 'initial excluded volume factor'),
    ('ev_stop', 0.0, float, 'final excluded volume factor'),
    ('ev_step', 0, int, 'If larger than zero, performs <n> rounds scaling '
                        'excluded volume factors from ev_start to ev_stop'),
    ('i', -1, int, 'Index of the structure, used for fish and single cell' 
                   'data'),
]


def validate_user_args(kwargs):
    args = {k: v for k, v, _, _ in ARG_DEFAULT}
    atypes = {k: t for k, _, t, _ in ARG_DEFAULT}
    for k, v in kwargs.items():
        if k not in args:
            raise ValueError('Keywords argument %s not recognized.' % k)
        if v is not None:
            args[k] = atypes[k](v)

    if args['write'] == -1:
        args['write'] = args['mdsteps']  # write only final step
    
    return args



def read_damid(damid):
    assert(damid is not None)
    if isinstance(damid, str) or isinstance(damid, unicode):
        if os.path.getsize(damid) > 0:
            damid_actdists = np.genfromtxt(damid,
                                           usecols=(0, 1, 2),
                                           dtype=(int, float, float))
            if damid_actdists.shape == ():
                damid_actdists = [damid_actdists]
        else:
            damid_actdists = []
    return damid_actdists


def create_lammps_data(model, user_args):
    
    n_atom_types = len(model.atom_types)
    n_bonds = len(model.bonds)
    n_bondtypes = len(model.bond_types)
    n_atoms = len(model.atoms)

    with open(user_args['data'], 'w') as f:

        print('LAMMPS input\n', file=f)

        print(n_atoms, 'atoms\n', file=f)
        print(n_atom_types, 'atom types\n', file=f)
        print(n_bondtypes, 'bond types\n', file=f)
        print(n_bonds, 'bonds\n', file=f)

        # keeping some free space to be sure
        print('-6000 6000 xlo xhi\n',
              '-6000 6000 ylo yhi\n',
              '-6000 6000 zlo zhi', file=f)

        print('\nAtoms\n', file=f)
        # index, molecule, atom type, x y z.
        for atom in model.atoms:
            print(atom, file=f)
        
        # bonds
        # Harmonic Upper Bond Coefficients are one for each bond type
        # and coded as:
        #   Bond_type_id kspring activation_distance
        # Each bond is coded as:
        #   Bond_id Bond_type_id ibead jbead

        if n_bonds > 0:
            print('\nBond Coeffs\n', file=f)
            for bt in model.bond_types:
                print(bt, file=f)

            print('\nBonds\n', file=f)
            for bond in model.bonds:
                print(bond, file=f)

        # Excluded volume coefficients
        atom_types = list(model.atom_types.values())

        print('\nPairIJ Coeffs\n', file=f)
        for i in range(len(atom_types)):
            a1 = atom_types[i]
            for j in range(i, len(atom_types)):
                a2 = atom_types[j]
                id1 = min(a1.id+1, a2.id+1)
                id2 = max(a1.id+1, a2.id+1)
                if (a1.atom_category == AtomType.BEAD and
                    a2.atom_category == AtomType.BEAD):
                    ri = a1.radius
                    rj = a2.radius
                    dc = (ri + rj)
                    A = (dc/math.pi)**2
                    #sigma = dc / 1.1224 #(2**(1.0/6.0))
                    #print(i+1, user_args['evfactor'], sigma, dc, file=f)
                    
                    print(id1, id2, A*user_args['evfactor'], dc, file=f)
                else:
                    print(id1, id2, 0.0, 0.0, file=f)
        

def create_lammps_script(model, user_args):
    maxrad = max([at.radius for at in model.atom_types if 
                  at.atom_category == AtomType.BEAD])

    with open(user_args['lmp'], 'w') as f:
        print('units                 lj', file=f)
        print('atom_style            bond', file=f)
        print('bond_style  hybrid',
              'harmonic_upper_bound',
              'harmonic_lower_bound', file=f)
        print('boundary              f f f', file=f)

        # Needed to avoid calculation of 3 neighs and 4 neighs
        print('special_bonds lj/coul 1.0 1.0 1.0', file=f)


        # excluded volume
        print('pair_style soft', 2.0 * maxrad, file=f)  # global cutoff

        print('read_data', user_args['data'], file=f)
        print('mass * 1.0', file=f)

        # groups atom types by atom_category
        sortedlist = list(sorted(model.atom_types, key=lambda x: x.atom_category))
        groupedlist = {k: list(v) for k, v in groupby(sortedlist, 
                                                key=lambda x: x.atom_category)}


        bead_types = [str(x) for x in groupedlist[AtomType.BEAD]]
        dummy_types = [str(x) for x in groupedlist[AtomType.FIXED_DUMMY]]
        centroid_types = [str(x) for x in groupedlist[AtomType.CLUSTER_CENTROID]]
        print('group beads type', ' '.join(bead_types), file=f)

        if dummy_types:
            print('group dummy type', ' '.join(dummy_types) , file=f)
            print('neigh_modify exclude group dummy all', file=f)
        if centroid_types:
            print('group centroid type', ' '.join(centroid_types), file=f)
            print('neigh_modify exclude group centroid all', file=f)

        print('group nonfixed type', ' '.join(centroid_types
                                            + bead_types), file=f)

        print('neighbor', maxrad, 'bin', file=f)  # skin size
        print('neigh_modify every 1 check yes', file=f)
        print('neigh_modify one', user_args['max_neigh'],
              'page', 20 * user_args['max_neigh'], file=f)

        
        # Freeze dummy atom
        if dummy_types:
            print('fix 1 dummy setforce 0.0 0.0 0.0', file=f)

        # Integration
        # select the integrator
        print('fix 2 nonfixed nve/limit', user_args['max_velocity'], file=f)
        # Impose a thermostat - Tstart Tstop tau_decorr seed
        print('fix 3 nonfixed langevin', user_args['tstart'], user_args['tstop'],
              user_args['damp'], user_args['seed'], file=f)
        print('timestep', user_args['timestep'], file=f)

        # Region
        # print('region mySphere sphere 0.0 0.0 0.0',
        #       user_args['nucleus_radius'] + 2 * maxrad, file=f)
        # print('fix wall beads wall/region mySphere harmonic 10.0 1.0 ',
        #       2 * maxrad, file=f)

        #print('pair_modify shift yes mix arithmetic', file=f)

        # outputs:
        print('dump   id beads custom',
              user_args['write'],
              user_args['out'],
              'id type x y z fx fy fz', file=f)

        # Thermodynamic info style for output
        print('thermo_style custom step temp epair ebond', file=f)
        print('thermo', user_args['thermo'], file=f)

        # ramp excluded volume
        if user_args['ev_step'] > 1:
            ev_val_step = float(user_args['ev_stop'] - user_args['ev_start']) / (user_args['ev_step'] - 1)
            for step in range(user_args['ev_step']):
                print('variable evprefactor equal ',
                      user_args['ev_start'] + ev_val_step*step,
                      file=f)
                #'ramp(%f,%f)' % (user_args['ev_start'], user_args['ev_stop']),
                print('fix %d all adapt 0',
                      'pair soft a * * v_evprefactor scale yes',
                      'reset yes' % 4 + step, file=f)
                print('run', user_args['mdsteps'], file=f)
                print('unfix 4', file=f)
        else:
        # Run MD
            print('run', user_args['mdsteps'], file=f)

        # Run CG
        print('min_style cg', file=f)
        print('minimize', user_args['etol'], user_args['ftol'],
              user_args['max_cg_iter'], user_args['max_cg_eval'], file=f)


def generate_input(crd, radii, index, **kwargs):
    '''
    This function generate two input files for
    the LAMMPS executable.

    Parameters
    ----------
        crd : np.ndarray
            Starting bead coordinates
        radii : np.ndarray or list of floats
            Radius of each bead in the model
        index : alabtools.Index
            alabtools.Index instance for the system
        
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

    :Returns:
        *bond_list (lammps_hic.lammps.BondContainer)*
            List of all added bonds  
    '''
    assert (len(index) == len(crd) == len(radii))

    n_genome_beads = len(index)

    args = validate_user_args(kwargs)

    model = LammpsModel()
    
    # create an atom for each genome bead
    for i in range(n_genome_beads):
        atype = DNABead(radii[i])
        model.add_atom(atom_type=atype, xyz=crd[i])

    # apply restraints
    apply_nuclear_envelope_restraints(model, crd, radii, index, args)
    
    apply_consecutive_beads_restraints(model, crd, radii, index, args)
    
    if args['actdist'] is not None:
        apply_hic_restraints(model, crd, radii, index, args)

    if args['damid'] is not None:
        apply_damid_restraints(model, crd, radii, index, args)

    if args['fish'] is not None:
        apply_fish_restraints(model, crd, radii, index, args)

    if args['bc_cluster'] is not None:
        apply_barcoded_cluster_restraints(model, crd, radii, index, args)

    create_lammps_data(model, args)
    create_lammps_script(model, args)

    return model


def _check_violations(bond_list, crd):
    violations = Violations()
    for bond_no, bond in enumerate(bond_list.bonds):
        bt = bond_list.bond_types[bond.bond_type]
        d = norm(crd[bond.i] - crd[bond.j])
        if bt.type_str == 'harmonic_upper_bound':
            if d > bt.r0:
                absv = d - bt.r0
                relv = (d - bt.r0) / bt.r0
                if relv > 0.05:
                    violations.add(bond.i, bond.j, absv, relv, bond_list.bond_annotations[bond_no])
        if bt.type_str == 'harmonic_lower_bound':
            if d < bt.r0:
                absv = bt.r0 - d
                relv = (bt.r0 - d) / bt.r0
                if relv > 0.05:
                    violations.add(bond.i, bond.j, absv, relv, bond_list.bond_annotations[bond_no])
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


def mark_complete(i, fname):
    '''
    marks a minimization as completed
    '''
    dtype = np.int32
    with open(fname, 'r+b') as f:
        f.seek(i*dtype.itemsize)
        np.int32(1).tofile(f)


def lammps_minimize(crd, radii, index, run_name, tmp_files_dir='/dev/shm', log_dir='.', check_violations=True, **kwargs):
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
        system = generate_input(crd, radii, index, **kwargs)

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
        info['n_restr'] = len(system.bonds)
        info['n_hic_restr'] = len([1 for b in system.bonds 
                                   if b.restraint_type == BT.HIC])
        
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
            violations = _check_violations(system, new_crd)
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
        n_violated = 0
        for i, r in completed:
            n_violated += 1 

        logger.info('%d structures with violations over %d structures',
                     n_violated,
                     len(completed))
        
        write_hss(prefix + '.hss', new_crd, radii, chrom)
        np.savetxt(prefix + '_energies.dat', energies)
        return completed
    
    except:
        logger.error('bulk_minimization() failed', exc_info=True)
        raise

    
def _serial_lammps_call_net(iargs):
    '''
    Serial function to be mapped in parallel. //
    It is intended to be used only internally by parallel routines.

    Parameters
    ---------- 
        iargs (tuple): contains the following items:
            - fname_from (string): the path to the memory map containing the
                original coordinates
            - fname_to (string): the path to the memory map containing the 
                final coordinates
            - fname_param (string): the path to the parameters file
            - struct_id (int): the id of the structure to perform the
                minimization on
    
    Returns
    -------
        oargs (tuple): contains the following items:
            - returncode (int): 0 if success, None otherwise
            - info (dict): dictionary of info returned by lammps_minimize. 
                If the run fails, info is set to the formatted traceback
                string
            - n_violations (int): Number of violated restraint at the end of 
                minimization. 
    '''

    try:
        # importing here so it will be called on the parallel workers
        import json
        import traceback
        from .network_coord_io import CoordClient
        from alabtools.utils import HssFile
        
        # unpack arguments
        param_fname, struct_id = iargs
        with open(param_fname) as f:
            kwargs = json.load(f)
        
        input_hss = kwargs.pop('input_hss')
        input_crd = kwargs.pop('input_crd')
        output_crd = kwargs.pop('output_crd')
        run_name = kwargs.pop('run_name')
        status_annotations = '.' + output_crd + '.status'

        # read file
        with HssFile(input_hss, 'r') as f:
            radii = f.radii
            index = f.index

        with CoordClient(input_crd) as f:
            crd = f.get_struct(i)
        
        # perform minimization
        new_crd, info, violations = lammps_minimize(crd, radii, index, run_name, **kwargs)
        
        with CoordClient(output_crd) as f:
            f.set_struct(i, new_crd)

        mark_complete(i, status_annotations)

        return (0, info, len(violations))

    except:
        import sys
        etype, value, tb = sys.exc_info()
        infostr = ''.join(traceback.format_exception(etype, value, tb))
        return (None, infostr, 0)



def _serial_lammps_call(iargs):
    '''
    Serial function to be mapped in parallel. //
    It is intended to be used only internally by parallel routines.

    Parameters
    ---------- 
        iargs (tuple): contains the following items:
            - fname_from (string): the path to the memory map containing the
                original coordinates
            - fname_to (string): the path to the memory map containing the 
                final coordinates
            - fname_param (string): the path to the parameters file
            - struct_id (int): the id of the structure to perform the
                minimization on
    
    Returns
    -------
        oargs (tuple): contains the following items:
            - returncode (int): 0 if success, None otherwise
            - info (dict): dictionary of info returned by lammps_minimize. 
                If the run fails, info is set to the formatted traceback
                string
            - n_violations (int): Number of violated restraint at the end of 
                minimization. 
    '''

    try:
        # importing here so it will be called on the parallel workers
        import json
        import traceback
        import os
        from os.path import splitext, basename
        from .myio import read_hms, write_hms, HtfFile
        from alabtools.utils import HssFile
        from lazyio import PopulationCrdFile
        
        # unpack arguments
        xin, param_fname, xout, run_name = iargs
        
        if isinstance(xin, str):
            # for bw compatibility
            crd, radii, chrom, n_struct, n_bead = read_hms(xin)

        elif isinstance(xin, list) or isinstance(xin, tuple):
            fname = xin[0]
            base, ext = splitext(basename(fname))
            if ext == '.hts':
                grp = xin[1]
                with HtfFile(fname, 'r') as f:
                    crd = f.get_coordinates(grp)
                    radii = f.radii
                    chrom = f.index.chrom
            elif ext == '.bindat':
                hssf = xin[1]
                i = xin[2]
                with HssFile(hssf, 'r') as f:
                    radii = f.radii
                    chrom = f.index.chrom 
                with PopulationCrdFile(fname, 'r') as f:
                    crd = f.get_struct(i)
            else:
                raise ValueError('Unknown input format')
        else:
            raise ValueError('Wrong argument format')

        # read parameters from disk
        with open(param_fname) as f:
            kwargs = json.load(f)

        # perform minimization
        new_crd, info, violations = lammps_minimize(crd, radii, chrom, run_name, **kwargs)
        
        # write coordinates to disk asyncronously and return run info
        #ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        #ex.submit(write_hms, new_fname, new_crd, radii, chrom, violations, info)
        if isinstance(xout, str):
            write_hms(xout, new_crd, radii, chrom, violations, info)
            os.chmod(xout, 0o664)
        elif isinstance(xout, list) or isinstance(xout, tuple):
            fname = xout[0]
            base, ext = splitext(basename(fname))
            if ext == '.hts':
                grp = xout[1]
                with HtfFile(fname) as f:
                    try:
                        f[grp]
                    except KeyError:
                        f.create_group(grp)
                    f.set_coordinates(grp, new_crd)
                    f.set_violations(grp, violations)
                    f.set_info(grp, info)
            elif ext == '.bindat':
                i = xout[1]
                with PopulationCrdFile(fname) as f:
                    f.set_struct(i, new_crd)
            else:
                raise ValueError('Unknown output format')
        else:
            raise ValueError('Wrong argument format')
            


        return (0, info, len(violations))

    except:
        import sys
        etype, value, tb = sys.exc_info()
        infostr = ''.join(traceback.format_exception(etype, value, tb))
        return (None, infostr, 0)

    

def bulk_minimize_single_file(parallel_client,
                              input_hss,
                              input_crd,
                              output_crd,
                              n_struct,
                              workdir='.',
                              tmp_files_dir='/dev/shm',
                              log_dir='.',
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
            logger.info('bulk_minimize_single_file(): ignoring completed runs.')
        else:
            for i in range(n_struct):

                # fname = '{}_{}.hms'.format(new_prefix, i)
                # fpath = os.path.join(workdir, fname)
                # if os.path.isfile(fpath):
                #     completed.append(i)
                # else:
                #     to_minimize.append(i)

                fname = 'copy%d.hts' % i
                fpath = os.path.join(workdir, fname)
                with HtfFile(fpath, 'r') as f:
                    if new_prefix not in f:
                        to_minimize.append(i) 

            if len(to_minimize) == 0:
                logger.info('bulk_minimize_single_file(): minimization appears to be already finished.')
                return (0, 0)
            logger.info('bulk_minimize_single_file(): Found %d already minimized structures. Minimizing %d remaining ones', len(completed), len(to_minimize))

        # prepare arguments as a list of tuples (map wants a single argument for each call)
        pargs = []
        for i in to_minimize:
            run_name = '{}_{}'.format(
                new_prefix,
                i
            )

            xin = (
                os.path.join(workdir, 'copy%d.hts' % i), 
                old_prefix,
            )

            xout = (
                os.path.join(workdir, 'copy%d.hts' % i), 
                new_prefix,
            )

            pargs.append((
                xin,
                parameter_fname,
                xout,
                run_name
            ))

        # using ipyparallel to map functions to workers
        engine_ids = list(parallel_client.ids)
        lbv = parallel_client.load_balanced_view()

        logger.info('bulk_minimize_single_file(): Starting bulk minimization of %d structures on %d workers', len(to_minimize), len(engine_ids))
        ar = lbv.map_async(_serial_lammps_call, pargs)
        logger.debug('bulk_minimize_single_file(): map_async sent.')

        # monitor progress
        monitor_progress('bulk_minimize_single_file() - %s' % new_prefix, ar)

        # post-analysis
        logger.info('bulk_minimize_single_file(): Done (Total time: %s)' % pretty_tdelta(ar.wall_time))
        results = list(ar.get())  # returncode, info, n_violations  

        # check for errors
        errors = [ (i, r[1]) for i, r in zip(to_minimize, results) if r[0] is None]
        now_completed = [ (i, r) for i, r in zip(to_minimize, results) if r[0] is not None]
        if len(errors):
            for i, e in errors:
                logger.error('Exception returned in minimizing structure %d: %s', i, e)
                print()
            raise RuntimeError('Unable to complete bulk_minimize_single_file()')
        
        # print a summary
        n_violated = sum([1 for i, r in now_completed if r[2] > 0 ])
        logger.info('%d structures with violations over %d structures',
                     n_violated,
                     len(now_completed))

        return len(now_completed), n_violated
    
    except:
        logger.error('bulk_minimization() failed', exc_info=True)
        raise



def bulk_minimize_single_file_net(parallel_client,
                                  input_hss,
                                  input_crd,
                                  output_crd,
                                  n_struct,
                                  workdir='.',
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
        kwargs['ignore_restart'] = ignore_restart
        kwargs['workdir'] = os.path.abspath(workdir)
        parameter_fname = os.path.join(workdir, new_prefix + '.parms.json')
        with open(parameter_fname, 'w') as fp:
            json.dump(kwargs, fp)
    
        # check if we already have some output files written.
        completed = []
        to_minimize = []
        if ignore_restart:
            to_minimize = list(range(n_struct))
            logger.info('bulk_minimize_single_file(): ignoring completed runs.')
        else:
            for i in range(n_struct):

                # fname = '{}_{}.hms'.format(new_prefix, i)
                # fpath = os.path.join(workdir, fname)
                # if os.path.isfile(fpath):
                #     completed.append(i)
                # else:
                #     to_minimize.append(i)

                fname = 'copy%d.hts' % i
                fpath = os.path.join(workdir, fname)
                with HtfFile(fpath, 'r') as f:
                    if new_prefix not in f:
                        to_minimize.append(i) 

            if len(to_minimize) == 0:
                logger.info('bulk_minimize_single_file(): minimization appears to be already finished.')
                return (0, 0)
            logger.info('bulk_minimize_single_file(): Found %d already minimized structures. Minimizing %d remaining ones', len(completed), len(to_minimize))

        # prepare arguments as a list of tuples (map wants a single argument for each call)
        pargs = []
        for i in to_minimize:
            run_name = '{}_{}'.format(
                new_prefix,
                i
            )

            xin = (
                os.path.join(workdir, 'copy%d.hts' % i), 
                old_prefix,
            )

            xout = (
                os.path.join(workdir, 'copy%d.hts' % i), 
                new_prefix,
            )

            pargs.append((
                xin,
                parameter_fname,
                xout,
                run_name
            ))

        # using ipyparallel to map functions to workers
        engine_ids = list(parallel_client.ids)
        lbv = parallel_client.load_balanced_view()

        logger.info('bulk_minimize_single_file(): Starting bulk minimization of %d structures on %d workers', len(to_minimize), len(engine_ids))
        ar = lbv.map_async(_serial_lammps_call, pargs)
        logger.debug('bulk_minimize_single_file(): map_async sent.')

        # monitor progress
        monitor_progress('bulk_minimize_single_file() - %s' % new_prefix, ar)

        # post-analysis
        logger.info('bulk_minimize_single_file(): Done (Total time: %s)' % pretty_tdelta(ar.wall_time))
        results = list(ar.get())  # returncode, info, n_violations  

        # check for errors
        errors = [ (i, r[1]) for i, r in zip(to_minimize, results) if r[0] is None]
        now_completed = [ (i, r) for i, r in zip(to_minimize, results) if r[0] is not None]
        if len(errors):
            for i, e in errors:
                logger.error('Exception returned in minimizing structure %d: %s', i, e)
                print()
            raise RuntimeError('Unable to complete bulk_minimize_single_file()')
        
        # print a summary
        n_violated = sum([1 for i, r in now_completed if r[2] > 0 ])
        logger.info('%d structures with violations over %d structures',
                     n_violated,
                     len(now_completed))

        return len(now_completed), n_violated
    
    except:
        logger.error('bulk_minimization() failed', exc_info=True)
        raise
