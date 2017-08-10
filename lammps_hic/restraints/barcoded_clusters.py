import numpy as np
import h5py
from ..lammps_utils import BT, BS, Atom

def get_cluster_size(n, radii):
    '''
    Determine the radius of the sphere containing the cluster

    Parameters
    ----------
        n : int 
            number of beads in the cluster
        radii : np.ndarray(dtype=float)
            radii of the beads

    Returns
    -------
        float : the radius of the cluster restraint sphere
    '''
    return 2 * np.sqrt(n - 1) * np.average(radii)


def apply_barcoded_cluster_restraints(system, coord, radii, cluster_file, user_args):
    struct_i = user_args['i']
    with h5py.File(cluster_file) as f:
        idxptr = f['idxptr']['()']
        structidx = f['structidx']['()']
        (curr_clusters, ) = np.where(structidx == struct_i)
        for cluster_id in curr_clusters:
            start_pos = idxptr[cluster_id]
            end_pos = idxptr[cluster_id + 1]
            beads = f['data'][start_pos:end_pos][()]
            csize = get_cluster_size(len(beads), radii[beads])
            
            # add a centroid for the cluster
            centroid_pos = np.mean(coord[beads], axis=0)
            centroid = system.add_atom(centroid_pos, radius=0.0) # no excluded volume
            
            parms = {
                'style' : BS.HARMONIC_UPPER_BOUND,
                'k' : 1.0,
                'r' : csize,
            }

            for b in beads:
                system.add_bond(centroid, system.atoms[b], 
                                parms, BT.BARCODED_CLUSTER)    
            
