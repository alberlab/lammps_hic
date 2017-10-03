#!/usr/bin/env python

import argparse

from lammps_hic.actdist import compute_activation_distances, read_actdist_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate activation distances')
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-n', '--name', help='ID of the run', 
                               required=True)
    required_args.add_argument('-H', '--hss', help='hss file containing the index for the system', 
                               required=True)
    required_args.add_argument('-c', '--crd', help='binary population coordinates file', 
                               required=True)
    required_args.add_argument('-o', '--out', help='output actdist file',     
                               required=True) 
    required_args.add_argument('-p', '--pmat', help='target probability matrix', 
                               required=True) 
    required_args
    required_args.add_argument('-s', '--sigma', help='cutoff frequency in 0-1 range', 
                               type=float, required=True)

    opt_args = parser.add_argument_group('optional arguments')
    opt_args.add_argument('--last', help='previous iteration actdist file. Set None',
                          type=str, default=None)
    opt_args.add_argument('--log', help='log to this file - defaults to the <run_ID>.log',
                          type=str)
    opt_args.add_argument('--serial-batch-size', help='number of bead pairs to be processed in a single serial task',
                          type=int, default=100)
    opt_args.add_argument('--parallel-batch-size', help='number of serial tasks to be mapped'\
                          ' in a task batch. This is a workaround for a memory leak in ipyparallel.'\
                          '', type=int, default=2000)
    opt_args.add_argument('--ignore-restart', help='if specified, overwrite previously generated run files',
                          action='store_true')
    opt_args.add_argument('--max-memory', help='maximum memory assigned to coordinate manager on workers',
                          type=str, default='2GB')

    args = parser.parse_args()

    if args.last is None:
        last = []
    else:
        last = read_actdist_file(args.last)

    compute_activation_distances(args.name, args.hss,args.crd, 
                                 args.pmat, args.sigma, args.out, last, 
                                 args.log, args.serial_batch_size, 
                                 args.parallel_batch_size, args.ignore_restart,
                                 args.max_memory) 
    



