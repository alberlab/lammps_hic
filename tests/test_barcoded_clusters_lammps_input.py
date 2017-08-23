import numpy as np
import h5py
import tempfile
import os
from alabtools.utils import Index
from lammps_hic.lammps import generate_input


def test_cluster_restraints():
    crds = np.array([
        [
            [ 50., 0., 0.],
            [100., 0., 0.],
            [0.,    0., 100.],
            [0.,  100.,   0.],
            [0., -100.,   0.]
        ],
        [
            [ 50., 0., 0.],
            [100., 0., 0.],
            [0.,    0., 60.],
            [0.,  100.,  0.],
            [0., -100.,  0.]
        ],
    ])
    radii = np.array([10.0] * len(crds[0]))
    index = Index([0]*5, [i*10 for i in range(5)], [i*10 - 1 for i in range(1,6)])

    clusters_idx = [0, 1, 2, 3, 4]
    clusters_assignment = [1, 1]
    clusters_idxptr = [0, 2, 5] 

    h, h5fn = tempfile.mkstemp()

    with h5py.File(h5fn, 'w') as h5f:
        h5f.create_dataset('data', data=clusters_idx, dtype='int')
        h5f.create_dataset('assignment', data=clusters_assignment, dtype='int')
        h5f.create_dataset('idxptr', data=clusters_idxptr, dtype='int')

    args = {'bc_cluster': h5fn, 'bc_cluster_size': 3, 'i': 1, 'lmp': 'lmp.tmp', 
            'data': 'input.tmp', 'mdsteps': 100, 'seed': 62067354}

    generate_input(crds[1], radii, index, **args)

    expected_lmp = """units                 lj
atom_style            bond
bond_style  hybrid harmonic_upper_bound harmonic_lower_bound
boundary              f f f
special_bonds lj/coul 1.0 1.0 1.0
pair_style soft 20.0
read_data input.tmp
mass * 1.0
group beads type 1
group dummy type 2
neigh_modify exclude group dummy all
group centroid type 3
neigh_modify exclude group centroid all
group nonfixed type 3 1
neighbor 10.0 bin
neigh_modify every 1 check yes
neigh_modify one 2000 page 40000
fix 1 dummy setforce 0.0 0.0 0.0
fix 2 nonfixed nve/limit 5.0
fix 3 nonfixed langevin 20.0 1.0 50.0 62067354
timestep 0.25
dump   id beads custom 100 out.lammpstrj id type x y z fx fy fz
thermo_style custom step temp epair ebond
thermo 1000
run 100
min_style cg
minimize 0.0001 1e-06 500 500
"""
    expected_data = """LAMMPS input

8 atoms

3 atom types

4 bond types

14 bonds

-6000 6000 xlo xhi
 -6000 6000 ylo yhi
 -6000 6000 zlo zhi

Atoms

1 1 1 50.0 0.0 0.0
2 1 1 100.0 0.0 0.0
3 1 1 0.0 0.0 60.0
4 1 1 0.0 100.0 0.0
5 1 1 0.0 -100.0 0.0
6 1 2 0.0 0.0 0.0
7 1 3 75.0 0.0 0.0
8 1 3 0.0 0.0 20.0

Bond Coeffs

3 harmonic_upper_bound 1.0 8.17120592832
2 harmonic_upper_bound 1.0 40.0
4 harmonic_upper_bound 1.0 10.8008382305
1 harmonic_upper_bound 1.0 4990.0

Bonds

1 1 1 6
2 1 2 6
3 1 3 6
4 1 4 6
5 1 5 6
6 2 1 2
7 2 2 3
8 2 3 4
9 2 4 5
10 3 7 1
11 3 7 2
12 4 8 3
13 4 8 4
14 4 8 5

PairIJ Coeffs

3 3 0.0 0.0
2 3 0.0 0.0
1 3 0.0 0.0
2 2 0.0 0.0
1 2 0.0 0.0
1 1 40.5284734569 20.0
"""

    with open('lmp.tmp') as f:
        assert f.read() == expected_lmp 

    with open('input.tmp') as f:
        assert f.read() == expected_data

    os.remove('lmp.tmp')
    os.remove('input.tmp')
    os.remove(h5fn)



