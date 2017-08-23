import numpy as np

class BondType(object):
    '''
    Bond Type. Abstract base class
    '''
    HARMONIC_UPPER_BOUND = 0
    HARMONIC_LOWER_BOUND = 1

    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError('This is an abstract base class')

    def __hash__(self):
        raise NotImplementedError('This is an abstract base class')

    def __eq__(self, other):
        return hash(self) == hash(other)

class HarmonicUpperBound(BondType):
    '''
    Harmonic upper bound restraint. Force is computed as
    F = -k * (r-r0)     if r > r0
    F = 0               otherwise 
    '''
    style_str = 'harmonic_upper_bound'
    style_id = BondType.HARMONIC_UPPER_BOUND

    def __init__(self, b_id=-1, k=1.0, r0=0.0):
        self.id = b_id
        self.k = k
        self.r0 = r0

    def __str__(self):
        return '{} {} {} {}'.format(self.id + 1,
                                    self.__class__.style_str,
                                    self.k,
                                    self.r0)

    def __hash__(self):
        return hash((self.__class__.style_id, self.k, self.r0))



class HarmonicLowerBound(BondType):
    '''
    Harmonic lower bound restraint. Force is computed as
    F = -k * (r0 - r)     if r < r0
    F = 0               otherwise 
    '''
    style_str = 'harmonic_lower_bound'
    style_id = BondType.HARMONIC_LOWER_BOUND

    def __init__(self, b_id=-1, k=1.0, r0=0.0):
        self.id = b_id
        self.k = k
        self.r0 = r0

    def __str__(self):
        return '{} {} {} {}'.format(self.id + 1,
                                    self.__class__.style_str,
                                    self.k,
                                    self.r0)

    def __hash__(self):
        return hash((self.__class__.style_id, self.k, self.r0))

    #def


class Bond(object):
    '''
    A bond between two atoms. The indexes are saved in
    the *0, ... , N-1* index, while its string
    representation is in the *1, ..., N* LAMMPS
    format.
    '''
    OTHER=-1
    CONSECUTIVE = 0
    HIC = 1
    DAMID = 2
    FISH_RADIAL = 3
    FISH_PAIR = 4
    BARCODED_CLUSTER = 5
    ENVELOPE = 6

    def __init__(self, b_id, bond_type, i, j, restraint_type=OTHER):
        self.id = b_id
        self.bond_type = bond_type
        self.i = i
        self.j = j
        self.restraint_type=restraint_type

    def __str__(self):
        return '{} {} {} {}'.format(self.id + 1,
                                    self.bond_type.id + 1,
                                    self.i.id + 1,
                                    self.j.id + 1)

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__


class AtomType(object):
    '''
    Atom type. Abstract class.
    '''
    BEAD = 0
    CLUSTER_CENTROID = 1
    FIXED_DUMMY = 2

    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError('This is an abstract base class')

    def __hash__(self):
        raise NotImplementedError('This is an abstract base class')

    def __eq__(self, other):
        return hash(self) == hash(other)


class DNABead(AtomType):
    '''
    DNA beads are defined by their radius
    '''
    atom_category = AtomType.BEAD

    def __init__(self, radius, type_id=-1):
        self.id = type_id
        self.radius = radius

    def __str__(self):
        return str(self.id + 1)

    def __hash__(self):
        return hash((self.__class__.atom_category, self.radius))


class FrozenPhantomBead(AtomType):
    MAX_BONDS = 20
    atom_category = AtomType.FIXED_DUMMY
    def __init__(self, type_id=-1):
        self.id = type_id

    def __str__(self):
        return str(self.id + 1)

    def __hash__(self):
        return hash(self.__class__.atom_category)


class ClusterCentroid(AtomType):
    atom_category = AtomType.CLUSTER_CENTROID
    def __init__(self, type_id=-1):
        self.id = type_id

    def __str__(self):
        return str(self.id + 1)
        
    def __hash__(self):
        return hash(self.__class__.atom_category)


class Atom(object):
    BEAD = 0
    CLUSTER_CENTROID = 1
    FIXED_DUMMY = 2

    def __init__(self, a_id, atom_type, mol_id, xyz):
        self.id = a_id
        self.atom_type = atom_type
        self.mol_id = mol_id
        self.xyz = xyz
        self.nbonds = 0

    def __str__(self):
        return '{} {} {} {} {} {}'.format(self.id + 1, 
                                          self.mol_id + 1, 
                                          self.atom_type.id + 1, 
                                          self.xyz[0],
                                          self.xyz[1],
                                          self.xyz[2])


class LammpsModel(object):

    def __init__(self):
        self.atoms = []
        self.atom_types = {}
        self.bonds = []
        self.bond_types = {}
        self.nmol = 1
        
    def get_next_dummy(self, pos=np.array([0., 0., 0.])):
        if (self.atoms[-1].atom_type == Atom.FIXED_DUMMY and
            self.atoms[-1].nbonds < FrozenPhantomBead.MAX_BONDS and
            np.all(self.atoms[-1].xyz == pos)):
                dummy = self.atoms[-1]
        else:
            atype = FrozenPhantomBead()
            dummy = self.add_atom(atype, xyz=pos)

        return dummy

    def add_bond(self, i, j, bond_type, restraint_type=Bond.OTHER):
        if isinstance(i, int):
            i = self.atoms[i]
        if isinstance(j, int):
            j = self.atoms[j]
        btype = self.bond_types.get(bond_type, None)
        if btype is None:
            bond_type.id = len(self.bond_types)
            btype = self.bond_types[bond_type] = bond_type

        bond_id = len(self.bonds)
        bond = Bond(bond_id, btype, i, j, restraint_type)
        self.bonds.append(bond)
        i.nbonds += 1
        j.nbonds += 1
        return bond

    def add_atom(self, atom_type, 
                 mol_id=0, # molecules are not used anyway
                 xyz=np.array([0., 0., 0.])):
        
        atom_id = len(self.atoms)
        atype = self.atom_types.get(atom_type, None)
        if atype is None:
            atom_type.id = len(self.atom_types)
            atype = self.atom_types[atom_type] = atom_type
        atom = Atom(atom_id, atype, mol_id, xyz)
        self.atoms.append(atom)
        self.nmol = max(mol_id, self.nmol)
        return atom
        