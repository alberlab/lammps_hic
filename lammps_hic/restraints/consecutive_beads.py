from ..lammps_utils import BS, BT

def apply_consecutive_beads_restraints(system, index, radii, user_args):
    n_genome_beads = len(index)

    for i in range(n_genome_beads - 1):
        if index.chrom[i] == index.chrom[i + 1]:
            parms = {
                'k' : 1.0,
                'r' : 2*(radii[i]+radii[i+1]),
                'style' : BS.HARMONIC_UPPER_BOUND,
            }
            system.add_bond(system.atoms[i], system.atoms[i+1], parms, 
                            BT.CONSECUTIVE)
            