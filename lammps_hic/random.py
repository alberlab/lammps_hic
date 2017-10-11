#!/usr/bin/env python

# Copyright (C) 2016 University of Southern California and
#                        Guido Polles
# 
# Authors: Guido Polles
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
The **random** module provides functions for the generation
of the initial coordinates
'''

from __future__ import print_function, division
import numpy
import numpy.random
import logging
import time
from math import acos, sin, cos, pi

from .util import pretty_tdelta
from .network_coord_io import CoordServer
from .population_coords import PopulationCrdFile
from .parallel_controller import ParallelController

__author__  = "Guido Polles"
__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "polles@usc.edu"


def uniform_sphere(R):
    '''
    Generates uniformly distributed points in a sphere
    
    Arguments:
        R (float): radius of the sphere

    Returns:
        numpy.array:
            triplet of coordinates x, y, z 
    '''
    phi = numpy.random.uniform(0, 2 * pi)
    costheta = numpy.random.uniform(-1, 1)
    u = numpy.random.uniform(0, 1)

    theta = acos( costheta )
    r = R * ( u**(1./3.) )

    x = r * sin( theta) * cos( phi )
    y = r * sin( theta) * sin( phi )
    z = r * cos( theta )

    return numpy.array([x,y,z])


def generate_territories(index, R=5000.0):
    '''
    Creates a single random structure with chromosome territories.

    Each "territory" is a sphere with radius 0.75 times the average
    expected radius of a chromosome.

    Arguments:
        chrom : alabtools.utils.Index 
            the bead index for the system.
        R : float 
            radius of the cell
    
    Returns:
        numpy.array : structure coordinates
    '''
    
    # chromosome ends are detected when
    # the name is changed
    n_tot = len(index)
    n_chrom = len(index.chrom_sizes)
    
    crds = numpy.empty((n_tot, 3))
    # the radius of the chromosome is set as 75% of its
    # "volumetric sphere" one. This is totally arbitrary. 
    # Note: using float division of py3
    chr_radii = [0.75 * R * (float(nb)/n_tot)**(1./3) for nb in index.chrom_sizes]
    crad = numpy.average(chr_radii)
    k = 0
    for i in range(n_chrom):    
        center = uniform_sphere(R - crad)
        for j in range(index.chrom_sizes[i]):
            crds[k] = uniform_sphere(crad) + center
            k += 1

    return crds
    

def _write_random_to_server(i, fname, index, R=5000.0):
    from network_coord_io import CoordClient
    crd = generate_territories(index, R=R)
    with CoordClient(fname) as client:
        client.set_struct(i, crd)



def create_random_population_with_territories(path, index, n_struct, 
                                              ipp_client=None,
                                              max_memory='2GB'):
    '''
    Creates a population of `n_struct` structures, saving it to 
    a binary PopulationCrdFile.

    Every structure is generated by selecting the chromosome
    centroids, and positioning chromosome beads in a spherical volume 
    around the centroid.
    
    Parameters
    ----------
    path (str) : the path of the population crd file
    index (alabtools.utils.Index) : index for the system
    n_struct (int) : number of structures to generate
    ipp_client (ipyparallel.Client, optional): If None, will just produce the 
        population in a serial run.
        If set to a ipyparallel Client instance, will distribute the job to 
        the workers.
    '''
    logger = logging.getLogger(__name__)
    
    n_bead = len(index)

    if ipp_client is None:
        # serial run
        start = time.time()
        logger.info('Serial run started (%d structures)', n_struct)
            
        with PopulationCrdFile(path, mode='w', shape=(n_struct, n_bead, 3), 
                               max_memory=max_memory) as p:
            for i in range(n_struct):
                crd = generate_territories(index)
                p.set_struct(i, crd)

        end = time.time()
        logger.info('Serial run done. (timing: %s)', 
                    pretty_tdelta(end-start))

    else:
        # parallel run
        with CoordServer(path, mode='w', shape=(n_struct, n_bead, 3), 
                         max_memory=max_memory):
            pc = ParallelController(name='RandomPopulation',
                                    serial_fun=_write_random_to_server,
                                    args=range(n_struct),
                                    logfile='random.log')
            pc.set_const('fname', path)
            pc.set_const('index', index)
            pc.submit()
