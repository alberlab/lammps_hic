from ..lammps_utils import Bond, HarmonicUpperBound

def apply_nuclear_envelope_restraints(model, coord, radii, index, user_args):
    nuc_radius = user_args['nucleus_radius']
    for i in range(len(coord)):
        center = model.get_next_dummy()
        bt = HarmonicUpperBound(k=1.0, r0=nuc_radius-radii[i])
        model.add_bond(i, center, bt, restraint_type=Bond.ENVELOPE)