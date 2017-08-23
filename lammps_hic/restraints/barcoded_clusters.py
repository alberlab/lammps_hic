import numpy as np
import h5py
from ..lammps_utils import Bond, HarmonicUpperBound, ClusterCentroid

def cbrt(x):
        return (x)**(1./3.)

def get_cluster_size(radii, size_factor):
    return cbrt(size_factor * np.sum(radii**3))

def apply_barcoded_cluster_restraints(model, coord, radii, index, user_args):
    cluster_file = user_args['bc_cluster'] 
    size_factor = user_args['bc_cluster_size']
    struct_i = user_args['i']

    if struct_i < 0:
        raise ValueError('Barcoded cluster restraints require to specify'
                         'a structure')

    centroid_type = ClusterCentroid()

    with h5py.File(cluster_file) as f:
        idxptr = f['idxptr'][()]
        assignment = f['assignment'][()]
        (curr_clusters, ) = np.where(assignment == struct_i)
        for cluster_id in curr_clusters:
            start_pos = idxptr[cluster_id]
            end_pos = idxptr[cluster_id + 1]
            beads = f['data'][start_pos:end_pos][()]
            crad = radii[beads]
            ccrd = coord[beads]
            csize = get_cluster_size(crad, size_factor)
            
            # add a centroid for the cluster
            centroid_pos = np.mean(ccrd, axis=0)
            centroid = model.add_atom(centroid_type, xyz=centroid_pos) # no excluded volume

            for b in beads:
                bt = HarmonicUpperBound(k=1.0, r0=csize-radii[b])
                model.add_bond(centroid, b, bt, Bond.BARCODED_CLUSTER)    
            
