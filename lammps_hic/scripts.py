from ipyparallel import Client
import logging
import os
import os.path

from .actdist import get_actdists
from .lammps import bulk_minimize
from .random import get_random_coordinates
from .myio import write_hss, read_full_actdist



def get_initial_coordinates(n_beads, radii, chrom, n_struct):
    logging.info('Creating initial coordinates')
    crd = get_random_coordinates(n_beads, n_struct)
    write_hss('random.hss', crd, radii, chrom)

    rc = Client()

    logging.info('Imposing chromosome territories')
    bulk_minimize(rc, 
                  crd_fname='random.hss', 
                  prefix='chromosome_territory',
                  evfactor=0.01,
                  territories=1,
                  mdsteps=30000)

    logging.info('Relaxing conformations')
    bulk_minimize(rc, 
                  crd_fname='chromosome_territory.hss', 
                  prefix='starting_configuration',
                  mdsteps=30000)


def run_modeling(client,
                 initial_coordinates_hss,
                 thetas,
                 sub_iterations,
                 probability_matrix,
                 **kwargs):

    rc = Client()
    last_ad = []
    last_hss = initial_coordinates_hss

    if not os.path.isdir('Actdist'):
        os.mkdir('Actdist')

    for theta in thetas:
        for sub_iter in sub_iterations:

            new_prefix = 'p%.3f_%s' % (theta, sub_iter)

            ad_fname = 'Actdist/' + new_prefix + '.ActDist'
            if not os.path.isfile(ad_fname):
                new_ad = get_actdists(client,
                                      last_hss,
                                      probability_matrix,
                                      theta,
                                      last_ad,
                                      save_to=ad_fname)
            else:
                new_ad = read_full_actdist(ad_fname)

            if not os.path.isfile(new_prefix + '.hss'):
                bulk_minimize(rc,
                              last_hss,
                              prefix=new_prefix,
                              evfactor=0.1,
                              actdist=new_ad,
                              **kwargs)

            last_hss = new_prefix + '.hss'
            last_ad = new_ad

