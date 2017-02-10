import argparse
from ipyparallel import Client
from lammps_hic.lammps import ARG_DEFAULT, bulk_minimize
import logging


class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimize population using lammps', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--argfile', help='(input) read arguments from file', type=open, action=LoadFromFile)

    parser.add_argument('inhss', help='input hss file', type=str)
    parser.add_argument('--prefix', help='output prefix', type=str, default='minimize')
    parser.add_argument('--tmp-files-dir', help='temporary files uses by lammps', type=str, default='/dev/shm')
    parser.add_argument('--log-dir', help='where to save error logs if any', type=str, default='.')
    parser.add_argument('--no-check-violations', help='turns off violations check', action='store_true')
    parser.add_argument('--ignore-restart', help='ignore restart files', action='store_true')

    other = parser.add_argument_group(description='Other options')

    for k, v in ARG_DEFAULT.items():
        astr = '--' + k.replace('_','-')
        if v is None:
            atyp = str
        else:
            atyp = type(v)
        other.add_argument(astr, help=k, type=atyp, default=v)

    args = parser.parse_args()
    arg_dict = vars(args)

    parallel_client = Client()

    kwargs = {}

    for k, defv in ARG_DEFAULT.items():
        if arg_dict[k] != defv:
            kwargs[k] = arg_dict[k] 

    print (kwargs)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    bulk_minimize(parallel_client,
                  args.inhss,
                  prefix=args.prefix,
                  tmp_files_dir=args.tmp_files_dir,
                  log_dir=args.log_dir,
                  check_violations_=(not args.no_check_violations),
                  ignore_restart=args.ignore_restart,
                  **kwargs)
