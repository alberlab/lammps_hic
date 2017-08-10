from ..lammps_utils import *

    ##############################
    # Add chromosome territories #
    ##############################
def TODO():
    if args['territories']:
        # for the territories step we want to make chromosomes very dense (3x?)
        terr_occupancy = args['occupancy'] * 3
        
        chrom_start = 0
        for k in range(n_chrom):
            chrom_len = index.chrom_sizes[i]  # size is in beads
            chrom_end = chrom_start + chrom_len
            chr_radii = radii[chrom_start:chrom_end]
            d = 4.0 * np.mean(chr_radii)  # 4 times the average radius
            d *= (float(chrom_len) / terr_occupancy)**(1. / 3)
            
            parms = bond_parms.add({
                'k' : 1.0,
                'r' : d,
                'style' : BS.HARMONIC_UPPER_BOUND,
            })
            for i in range(chrom_start, chrom_end, args['territories']):
                for j in range(i + args['territories'], chrom_end,
                               args['territories']):
                    bonds.append((i, j, parms, BT.TERRITORY))

            chrom_start = chrom_end
