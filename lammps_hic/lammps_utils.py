import numpy as np

class BT(object):
    '''
    Restraint types
    '''
    NA = -1
    CONSECUTIVE = 0
    HIC = 1
    DAMID = 2
    FISH_RADIAL = 3
    FISH_PAIR = 4
    BARCODED_CLUSTER = 5
    TERRITORY = 6


class BS(object):
    '''
    Bond styles
    '''
    HARMONIC_UPPER_BOUND = 0
    HARMONIC_LOWER_BOUND = 1

type_string = {
    BS.HARMONIC_LOWER_BOUND : 'harmonic_lower_bound',
    BS.HARMONIC_UPPER_BOUND : 'harmonic_upper_bound',
}


class Atom(object):
    def __init__(self, crd, radius=1.0, frozen=False, molid=None, ID=None):
        self.crd = crd
        self.radius = radius
        self.frozen = frozen
        self.nbonds = 0
        self.id = ID
        self.molid = molid

class BondParameters(object):
    '''
    Bond specifications. On add() checks that the bond parameters have
    not already been specified.
    '''
    def __init__(self):
        self.hashes = {}
        self.parms = [] 
        self.n = 0

    def __len__(self):
        return self.n
    
    def add(self, parm):
        h = hash(tuple(sorted(parm.items())))
        pos = self.hashes.get(h, None)
        if pos is None:
            self.hashes[h] =  self.n
            self.n = self.n + 1
            self.parms.append(parm)
            return parm
        else:
            return self.parms[pos]

    def __getitem__(self, i):
        return self.parms[i]


class Bond(object):
    '''
    A bond between two atoms.
    '''
    def __init__(self, i, j, parameters, restraint_type=BT.NA):
        assert(i != j)
        i.nbonds += 1
        j.nbonds += 1
        self.i = i
        self.j = j
        self.bond_params = parameters
        self.restraint_type = restraint_type



DUMMY_MAX_BONDS = 20

class LammpsSystem(object):
    def __init__(self):
        self.bonds = []
        self.bond_parms = BondParameters()
        self.radius_to_atom_type = {} 
        self.atoms = []
    
    def get_next_dummy(self, pos=np.array([0., 0., 0.])):
        dummy = self.atoms[-1] 
        if not (dummy.frozen 
                and dummy.nbonds < DUMMY_MAX_BONDS
                and dummy.crd.array_equal(pos)):
            dummy = self.add_atom(pos, r=0.0, frozen=True)
        return dummy

    def add_bond(self, i, j, parms, bt):
        parms = self.bond_parms.add(parms)
        bond = Bond(i, j, parms, bt)
        self.bonds.append(bond)
        return bond

    def add_atom(self, *args, **kwargs):
        atom = Atom(*args, **kwargs)
        atom.id = len(self.atoms) + 1
        self.atoms.append(atom)
        return atom