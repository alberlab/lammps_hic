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
from .util import monitor_progress, pretty_tdelta, require_vars, resolve_templates, remove_if_exists
from .globals import lammps_executable, float_epsilon
from .restraints import *
from .lammps_utils import *
from .parallel_controller import ParallelController

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
        print('dump   crd_dump all custom',
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

def _check_violations(model, crd, i, tol=0.05):
    violations = []
    for bond in model.bonds:
        rv = bond.get_relative_violation(crd)
        if rv > tol:
            bt = bond.bond_type
            violations.append((i, bond.restraint_type, bond.i, bond.j,
                               bt.style_id, rv))
    return violations

def lammps_minimize(crd, radii, index, run_name, tmp_files_dir='/dev/shm', 
                    check_violations=True, keep_temporary_files=False,
                    **kwargs):
    '''
    Lammps interface for minimization.
    
    It first creates input and data files for lammps, then
    runs the lammps executable in a process (using subprocess.Popen).

    When the program returns, it parses the output and returns the 
    new coordinates, along informations on the run and on violations.

    The files created are 
    - input file: `tmp_files_dir`/`run_name`.lam
    - data file: `tmp_files_dir`/`run_name`.input
    - trajectory file: `tmp_files_dir`/`run_name`.lammpstrj

    The function tries to remove the temporary files after the run, both in 
    case of failure and success (unless `keep_temporary_files` is set to 
    `False`). If the interpreter is killed without being able to catch 
    exceptions (for example because of a walltime limit) some files could be 
    left behind.

    Parameters
    ----------
    crd : numpy.ndarray 
        Initial coordinates (n_beads x 3 array).
    radii : numpy.ndarray
        Radius for each particle in the system.
    index : `alabtools.Index`
        Index for the system.
    run_name : str
        Name of the run. It determines only the name of temporary files.
    tmp_files_dir : str
        Location of temporary files. Needs writing permissions. The default 
        value, /dev/shm, is usually a in-memory file system, useful to 
        share data between processes without the overhead of actually 
        writing to a physical disk.
    check_violations : bool
        Performs a check on the violations of the assigned bonds. If set 
        to `False`, the check is skipped and the violation output will be 
        an empty list.
    keep_temporary_files : bool
        If set to `True`, does not try to remove the temporary files 
        generated by the run.
    \*\*kwargs : dict 
        Optional keyword arguments for minimization. 
        See docs for `lammps.generate_input`.

    Returns
    -------
    new_crd : numpy ndarray 
        Coordinates after minimization.
    info : dict 
        Dictionary with summarized info for the run, as returned 
        by `lammps.get_info_from_log`.
    violations : list
        List of violations. If the `check_violations` parameter is set 
        to `False`, returns an empty list.

    Raises
    ------
    RuntimeError 
        If the lammps executable return code is different from 0, it raises 
        a RuntimeError with the contents of the standard error.
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


        if not keep_temporary_files:
            if os.path.isfile(data_fname):
                os.remove(data_fname)
            if os.path.isfile(script_fname):
                os.remove(script_fname)
            if os.path.isfile(traj_fname):
                os.remove(traj_fname)

    except:
        if not keep_temporary_files:
            if os.path.isfile(data_fname):
                os.remove(data_fname)
            if os.path.isfile(script_fname):
                os.remove(script_fname)
            if os.path.isfile(traj_fname):
                os.remove(traj_fname)
        raise

    if check_violations:
        violations = _check_violations(system, new_crd, i)
    else: 
        violations = []
    return new_crd, info, violations

    
def _setup_and_run_task(i):
    '''
    Serial function to be mapped in parallel. //
    It is a wrapper intended to be used only internally by the parallel map
    function.
    Checks global variables, resolve the templates, obtains input data,
    runs the minimization routines and finally communicates back results.

    Parameters
    ---------- 
    i : int
        number of the structure 
    
    '''

    # importing here so it will be called on the parallel workers
    from .network_coord_io import CoordClient
    from alabtools.utils import HssFile
    from .network_sqlite import SqliteClient
    
    # check that all the necessary varaibles have been set
    require_vars(['lammps_args', 'input_crd', 'output_crd', 'input_hss',
                  'str_templates', 'status_db', 
                  'check_violations', 'violations_db'])

    require_vars(['run_name'], str_templates)

    tmp_vars = resolve_templates(str_templates, [i])
    run_name = tmp_vars['run_name']

    # add i as a lammps argument
    lammps_args['i'] = i    

    # reads radii and index from hss
    with HssFile(input_hss, 'r') as f:
        radii = f.radii
        index = f.index

    # connects to the coordinates server
    with CoordClient(input_crd) as f:
        crd = f.get_struct(i)
    
    # connects to the status server
    status = SqliteClient(status_db)

    if check_violations:
        violations_cl = SqliteClient(violations_db)

    # perform minimization
    new_crd, info, violations = lammps_minimize(crd, radii, index, run_name, **lammps_args)
    
    # outputs coordinates
    with CoordClient(output_crd) as f:
        f.set_struct(i, new_crd)

    if check_violations and violations:
        violations_cl.executemany('INSERT INTO violations VALUES ' +
                                  '(?, ?, ?, ?, ?, ?)', violations)

    # set as complete 
    status.execute('INSERT INTO completed VALUES (?, ?, ?, ?, ?, ?)', 
                   (i, int(time.time()), info['final-energy'],
                    info['pair-energy']), info['bond-energy'],
                    info['md-time'])

def minimize_population(run_name, input_hss, input_crd, output_crd,
                        n_struct, workdir='.', tmp_files_dir='/dev/shm',
                        ignore_restart=False, log=None, lammps_args={},
                        check_violations=False, max_memory='2GB'):
    '''
    Uses ipyparallel to minimize a population of structures.

    Parameters
    ----------
        *parallel_client (ipyparallel.Client instance)* 
            The ipyparallel.Client instance on which

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

    logger = logging.getLogger('minimize_population()')
    logger.setLevel(logging.DEBUG)

    # Decide file names
    ndigit = len(str(n_struct))
    status_db = run_name + '.status.db'
    violations_db = run_name + '.violations.db'
    str_templates = {
        'run_name' = run_name + '-{0:0#d}'.replace('#',str(ndigit))
    }

    pc = ParallelController(name=run_name, 
                            logfile=log, 
                            serial_fun=_setup_and_run_task)

    # set the variables on the workers
    pc.set_global('lammps_args', lammps_args)
    pc.set_global('input_crd', input_crd)
    pc.set_global('output_crd', output_crd)
    pc.set_global('input_hss', input_hss)
    pc.set_global('str_templates', str_templates)
    pc.set_global('status_db', status_db)
    pc.set_global('violations_db', violations_db)
    pc.set_global('check_violations', check_violations)

    # prepare i/o servers
    violations_setup = ('CREATE TABLE violations (struct INT, '
        'restraint_type INT, i INT, j INT, style INT, rv REAL)')
    status_setup = ('CREATE TABLE completed (struct INT, '
        'timestamp INT, efinal REAL, epair REAL, ebond REAL, '
        'mdtime REAL)')

    incrd_srv = CoordServer(input_crd, mode='r', max_memory=max_memory)

    if ignore_restart:
        logger.info('Ignoring completed runs.')
        openmode = 'w'
    else:
        openmode = 'r+'
    
    outcrd_srv = CoordServer(output_crd, mode=openmode, shape=incrd_srv.shape, 
                             max_memory=max_memory)
    violations_srv = SqliteServer(violations_db, violations_setup, mode=openmode)
    status_srv = SqliteServer(status_db, status_setup, mode=openmode)

    incrd_srv.start()
    outcrd_srv.start()
    violations_srv.start()
    status_srv.start()
    
    # check if we have already completed tasks
    with SqliteClient(status_db) as clt:
        completed = clt.fetchall('SELECT struct from completed')
    to_process = [x for x in range(n_struct) if x not in completed]
    logger.info('%d jobs completed, %d jobs to be submitted.', 
                len(completed), len(to_process))

    # map the jobs to the workers
    pc.args = to_process
    pc.submit()

    logger.info('Run %s completed', run_name)

