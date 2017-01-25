from .actdist import get_actdists
from .lammps import lammps_minimize
from .myio import read_hss


def run_modeling(client, thetas, sub_iterations, **kwargs):

    last_ad = None

    for theta in thetas:
        for sub_iter in sub_iterations:
            
            new_prefix = 'p%.3f_%s' % (theta, sub_iter)
            ad_fname = 'Actdist/' + new_prefix + '.ActDist'
            new_ad = get_actdists(client, last_hss, pmat, theta, last_ad, save_to=ad_fname)

            


 
    if len(chrom) == crd.shape[0] // 2:
        chrom = list(chrom) + list(chrom)
