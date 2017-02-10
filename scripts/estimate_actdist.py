#!/usr/bin/env python

import argparse
import h5py

from ipyparallel import Client

from lammps_hic.actdist import get_actdists
from lammps_hic.myio import read_full_actdist, read_hss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate activation distances')
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--crd',     help='population hss file',    required=True)
    required_args.add_argument('--out',     help='output actdist file',     required=True) 
    required_args.add_argument('--pmat',    help='target probability matrix',           required=True) 
    required_args.add_argument('--last',    help='previous iteration actdist file',     default='None', required=True)
    required_args.add_argument('--freq',    help='cutoff frequency in 0-1 range',       type=float,     required=True)

    opt_args = parser.add_argument_group('optional arguments')
    opt_args.add_argument('--scatter', help='subdivide the work in about <scatter>^2 * n_workers chunks', type=int, default=10)


    args = parser.parse_args()

    crd, radii, chrom, n_struct, n_bead = read_hss(args.crd)

    with h5py.File(args.pmat, 'r') as pm:
        probmat    = pm['matrix'][()]
    n_loci = len(probmat)

    if args.last != 'None':
        last_ad = read_full_actdist(args.last)
    else:
        last_ad = None

    client = Client()

    get_actdists(client, args.crd, probmat, args.freq, last_ad=last_ad, save_to=args.out, scatter=args.scatter)

    



