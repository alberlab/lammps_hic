from lammps_hic.lammps import BondParameters
    
def test_answer():
    bp = BondParameters()
    bp.add({'k': 1.0, 's': 1, 'r': 30.0})
    bp.add({'k': 1.0, 's': 1, 'r': 30.0})
    assert(len(bp) == 1)
    bp.add({'k': 2.0, 's': 1, 'r': 30.0})
    assert(len(bp) == 2)
    assert(bp[1] == {'k': 2.0, 's': 1, 'r': 30.0})
