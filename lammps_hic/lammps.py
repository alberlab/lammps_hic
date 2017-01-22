from __future__ import print_function, division
import os
import os.path
import subprocess
from io import StringIO
import shutil
import h5py
import numpy as np
from numpy.linalg import norm


lammps_executable = 'lmp_serial_mod'


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
    'soft_min': 0,
    'max_neigh': 2000,
    'max_cg_iter': 500,
    'max_cg_eval': 500,
    'etol': 1e-4,
    'ftol': 1e-6,
    'territories': False,
}


class BondType(object):

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


class Bond(object):

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


class BondContainer(object):

    def __init__(self):
        self.bond_types = []
        self.bonds = []

    def add_type(self, type_str, kspring, r0):
        bt = BondType(len(self.bond_types),
                      type_str,
                      kspring,
                      r0)
        self.bond_types.append(bt)
        return bt

    def add_bond(self, bond_type, i, j):
        bond = Bond(len(self.bonds),
                    bond_type.id,
                    i,
                    j)
        self.bonds.append(bond)
        return bond


class DummyAtoms(object):
    '''this needs an explanation. To enforce distance from center,
    we use a bond with a dummy atom in the middle. Hovever, if we use
    the same atom for too many bonds, the dummy atom will set the
    max_bonds parameter for all atoms to a very large number,
    making memory usage explode. Hence, it's better to have multiple
    dummy atoms'''
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
    ''' transform a list of strings in numeric
    ids (from 1 to N) '''
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


def generate_input_single_radius(crd, bead_radius, chrom, **kwargs):
    ''' from coordinates of the beads, a single bead radius
    and the list of relative chrmosomes, it constructs the
    input files for running a minimization. Has various arguments,
    the ones in the ARG_DEFAULT dictionary. This is a quite long
    function, performing the parsing and the output. '''

    args = {k: v for k, v in ARG_DEFAULT.items()}
    for k, v in kwargs.items():
        if v is not None:
            args[k] = v

    if args['write'] is None:
        args['write'] = args['mdsteps']  # write only final step

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

    ###############################
    # Add consecutive beads bonds #
    ###############################

    bt = bond_list.add_type('harmonic_upper_bound',
                            kspring=1.0,
                            r0=4 * bead_radius)

    for i in range(n_atoms - 1):
        if chrom[i] == chrom[i + 1]:
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
            chrom_len = chrom_start[k + 1] - chrom_start[k]
            d = 2.0 * bead_radius
            d *= (float(chrom_len) / terr_occupancy)**(1. / 3)
            bt = bond_list.add_type('harmonic_upper_bound',
                                    kspring=1.0,
                                    r0=d)
            for i in range(chrom_start[k], chrom_start[k + 1]):
                for j in range(i + 1, chrom_start[k + 1]):
                    bond_list.add_bond(bt, i, j)

    ##############################################
    # Activation distance for contact restraints #
    ##############################################
    # NOTE: activation distances are surface to surface, not center to center
    # The old method was using minimum pairs
    # Here we enforce a single pair.
    if args['actdist'] is not None:
        if isinstance(args['actdist'], str):
            if os.path.getsize(args['actdist']) > 0:
                actdists = np.genfromtxt(args['actdist'],
                                         usecols=(0, 1, 3),
                                         dtype=(int, int, float))
            else:
                actdists = []

        for (i, j, d) in actdists:
            cd = np.zeros(4)
            rsph = bead_radius + bead_radius
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
        if isinstance(args['damid'], str):
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
            td = args['nucleus_radius'] - (2 * bead_radius)

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
    n_bead_types = 1
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
        for i, x in enumerate(x1):
            print(i + 1, chromid[i], 1, x[0], x[1], x[2], file=f)
        for i, x in enumerate(x2):
            print(i + 1 + n_hapl, chromid[i], 1, x[0], x[1], x[2], file=f)

        # dummy atoms in the middle
        dummy_mol = max(chromid) + 1
        for i in range(dummies.n_dummy):
            print(n_atoms + i + 1, dummy_mol, 2, 0, 0, 0, file=f)

        print('\nBond Coeffs\n', file=f)
        for bc in bond_list.bond_types:
            print(bc, file=f)

        print('\nBonds\n', file=f)
        for b in bond_list.bonds:
            print(b, file=f)

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
        print('pair_style lj/cut', 2 * bead_radius, file=f)  # global cutoff

        print('read_data', args['data'], file=f)
        print('mass * 1.0', file=f)

        print('group beads type !=', dummy_type, file=f)
        print('group dummy type', dummy_type, file=f)

        print('neighbor', 2 * bead_radius, 'bin', file=f)  # skin size
        print('neigh_modify every 10 check yes', file=f)
        print('neigh_modify one', args['max_neigh'],
              'page', 20 * args['max_neigh'], file=f)

        # Freeze dummy atom
        print('fix 1 dummy setforce 0.0 0.0 0.0', file=f)
        print('neigh_modify exclude group dummy all check no', file=f)

        # Integration
        # select the integrator
        print('fix 2 beads nve/limit', args['max_velocity'], file=f)
        # Impose a thermostat - Tstart Tstop tau_decorr seed
        print('fix 3 beads langevin', args['tstart'], args['tstop'],
              args['damp'], args['seed'], file=f)
        print('timestep', args['timestep'], file=f)

        # Region
        print('region mySphere sphere 0.0 0.0 0.0',
              args['nucleus_radius'] + 2 * bead_radius, file=f)
        print('fix wall beads wall/region mySphere harmonic 10.0 1.0 ',
              2 * bead_radius, file=f)

        # Initial minimization
        if args['soft_min'] != 0:
            print('pair_style soft', 2 * bead_radius, file=f)  # global cutoff
            print('pair_coeff 2 2 0.0', file=f)
            print('pair_coeff 1 1', args['evfactor'], 2 * bead_radius, file=f)
            print('run', args['soft_min'], file=f)
            print('pair_style lj/cut', 2 * bead_radius, file=f)  # reset

        # print('pair_coeff 2 2 0.0 1.0', file=f)
        print('pair_coeff * *', args['evfactor'], 2 * bead_radius / 1.122,
              file=f)

        print('pair_modify shift yes', file=f)

        # outputs:
        print('dump   id beads custom',
              args['write'],
              args['out'],
              'id type x y z fx fy fz', file=f)
        print('thermo', args['thermo'], file=f)

        # Run MD
        print('run', args['mdsteps'], file=f)

        # Run CG
        print('min_style cg', file=f)
        print('minimize', args['etol'], args['ftol'],
              args['max_cg_iter'], args['max_cg_eval'], file=f)


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
    ''' gets only the final energy.
    TODO: get more info? '''
    info = {}
    generator = _reverse_readline(output)
    for l in generator:
        if l[:9] == '  Force t':
            ll = next(generator)
            info['final-energy'] = float(ll.split()[2])
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


def lammps_minimize(i, last_hss, new_prefix, **kwargs):
    ''' lammps_minimize: calls lammps on files on
    /dev/shm for performance, and checks execution
    result and final energies '''

    data_fname = '/dev/shm/{}.{}.data'.format(new_prefix, i)
    script_fname = '/dev/shm/{}.{}.lam'.format(new_prefix, i)
    traj_fname = '/dev/shm/{}.{}.lam'.format(new_prefix, i)

    try:
        with h5py.File(last_hss, 'r') as f:
            crd = f['coordinates'][i, :, :][()]
            radius = f['radius'][0][()]
            chrom = f['idx'][:][()]

        # prepare input
        generate_input_single_radius(crd, radius, chrom, **kwargs)

        # run the lammps minimization
        output = StringIO()
        error = StringIO()
        with open(script_fname, 'r') as lamfile:
            return_code = subprocess.call([lammps_executable,
                                           '-log', '/dev/null'],
                                          stdin=lamfile,
                                          stdout=output,
                                          stderr=error)

        if return_code != 0:
            error_dump = './errs/{}.{}.lammps.err'.format(new_prefix, i)
            output_dump = './errs/{}.{}.lammps.log'.format(new_prefix, i)
            with open(error_dump, 'w') as fd:
                error.seek(0)
                shutil.copyfileobj(error, fd)

            with open(output_dump, 'w') as fd:
                output.seek(0)
                shutil.copyfileobj(output, fd)

            raise RuntimeError('LAMMPS exited with non-zero exit code')

        # get results
        info = get_info_from_log(output)
        with open(traj_fname, 'r') as fd:
            crd = get_last_frame(fd)

        os.remove(data_fname)
        os.remove(script_fname)
        os.remove(traj_fname)

    except:
        if os.path.isfile(data_fname):
            os.remove(data_fname)
        if os.path.isfile(script_fname):
            os.remove(script_fname)
        if os.path.isfile(traj_fname):
            os.remove(traj_fname)
        raise

    return crd, info
