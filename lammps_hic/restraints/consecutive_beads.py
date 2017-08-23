from ..lammps_utils import Bond, HarmonicUpperBound


def apply_consecutive_beads_restraints(model, coord, radii, index, user_args):
    
    n_genome_beads = len(index)
    for i in range(n_genome_beads - 1):
        if index.chrom[i] == index.chrom[i + 1]:
            rij = 2*(radii[i]+radii[i+1])
            bt = HarmonicUpperBound(r0=rij, k=1.0)
            model.add_bond(i, i+1, bt, 
                           restraint_type=Bond.CONSECUTIVE)
            