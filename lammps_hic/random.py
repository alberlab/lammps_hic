import numpy as np


def prepare_random_template(n_beads, nuclear_radius=5000.0):
    '''Creates a cubic grid of points inside the nuclear envelope.'''
    n1d = (float(n_beads)/0.52)**(1.0/3.0)
    a = (1.9 * nuclear_radius) / n1d  
    grid = np.concatenate([np.arange(0, -nuclear_radius, -a), np.arange(a, nuclear_radius, a)])
    crd = []
    r2 = nuclear_radius**2
    for x in grid:
        for y in grid:
            for z in grid:
                if x*x+y*y+z*z > r2:
                    continue
                crd.append(np.array([x,y,z]))
    return np.array(crd)
    

def get_random_coordinates(n_beads, n_conf, template_crds=None):
    '''Create a set of random starting configuration by randomly 
    assigning coordinates in a lattice inside the nuclear 
    envelope. In this way we don't face the situation where
    too many points are too close.'''
    if template_crds is None:
        template_crds = prepare_random_template(n_beads)
    crd = np.empty((n_conf, n_beads, 3))
    for i in range(n_conf):
        crd[i] = np.random.permutation(template_crds)[:n_beads]
    return crd