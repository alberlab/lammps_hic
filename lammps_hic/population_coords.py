import numpy as np
import os.path

def parseSize(size):
    units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    if isinstance(size, str):
        if size[-1].isdigit():
            return int(float(size))
        for i, s in enumerate(size):
            if not s.isdigit() and s != '.' and not s.isspace():
                break
        val = float(size[:i])
        unit = size[i:i+1].lower()

        return int(float(val)*units[unit])
    else:
        return size

class PopulationCrdFile(object):

    '''
    A file which contains coordinates for a population. 
    It performs lazy evaluations of "columns" (structure coordinates) and 
    "rows" (coordinates of a single bead across the population).
    Use only rows or only columns! This is not yet smart enough to update
    the other in-memory coordinates. Close and reopen to change from
    rows to columns and vice versa.
    The max_memory parameter controls the maximum amount of memory the object
    should use. Note that the memory is checked only on the raw data,
    does not take into consideration overhead for python structures and 
    internal variables.
    
    Parameters
    ----------
        fname : string
            Path of the file on disk
        mode : string
            Either 'r' (readonly), 'w' (create/truncate), or 'r+' (read
            and write)
        shape : list
            Dimensions of the population coordinates (n_struct x n_bead x 3).
            Ignored if in readonly/rw modes.
        dtype : numpy.dtype or string
            Data type. Defaults to 'float32'
        max_memory : int
            Maximum number of bytes to use in memory. Note that does not 
            take into consideration data structures overheads and similar.
            Will use slightly more memory than this maximum.
    '''

    def __init__(self, fname, mode='r+', shape=(0, 0, 3), dtype='float32', max_memory='2GB'):
        
        assert mode == 'r+' or mode == 'w' or mode =='r'
        max_memory = parseSize(max_memory)

        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.datasize = self.dtype.itemsize * np.prod(shape)

        if mode == 'r+' and not os.path.isfile(fname):
            mode = 'w'
        
        self.fd = open(fname, mode + 'b')
        if (mode == 'w'):
            assert len(shape) == 3
            self.shape = shape
            self.headersize = 4 + 4*(3)
            if (self.datasize > 0):
                self.fd.seek(self.datasize+self.headersize-1)
                self.fd.write(b"\0")
            self._write_header()
        else:
            ndim = np.fromfile(self.fd, dtype=np.int32, count=1)
            assert ndim == 3, 'file %s fname has dimension %d != 3' % (fname, ndim)
            self.shape = np.fromfile(self.fd, dtype=np.int32, count=3)
            self.headersize = 4 + 4*(3)

        self.nstruct = self.shape[0]
        self.nbead = self.shape[1]

        self.crd3size = self.itemsize * 3
        self.structsize = self.crd3size * self.nbead
        self.beadsize =  self.crd3size * self.nstruct


        self.beads = {}
        self.structs = {}

        self.cq = []
        self.bq = []
        self._cwrt = {}
        self._bwrt = {}
        self.max_memory = max_memory

        assert self.structsize <= max_memory and self.beadsize <= max_memory

    def __enter__(self):
        return self

    def used_memory(self):
        return (self.structsize * len(self.structs) + 
                self.beadsize * len(self.beads))

    def dump_coord(self, idx):
            self.fd.seek(self.headersize + self.structsize * idx)
            self.structs[idx].tofile(self.fd)

    def free_coord(self, idx):
        if idx in self._cwrt:
            self.dump_coord(idx)
            del self._cwrt[idx]
        del self.structs[idx]

    def dump_bead(self, idx):
        v = self.beads[idx]
        for i in range(self.nstruct):
            self.fd.seek(self.headersize + i * self.structsize + 
                         idx * self.crd3size)
            v[i].tofile(self.fd)

    def free_bead(self, idx):
        if idx in self._cwrt:
            self.dump_bead(idx)
            del self._bwrt[idx]       
        del self.beads[idx]

    def free_space(self, requested):
        assert requested <= self.max_memory
        while self.used_memory() + requested > self.max_memory:
            if len(self.structs):
                ir = self.cq.pop()
                self.free_coord(ir)
            else:
                ir = self.bq.pop()
                self.free_bead(ir)

    def get_struct(self, idx):
        idx = int(idx)
        if idx in self.structs:
            pos = self.cq.index(idx)
            self.cq.insert(0, self.cq.pop(pos))
            return self.structs[idx]
        else:
            self.free_space(self.structsize)
            self.fd.seek(self.headersize + self.structsize * idx)
            self.structs[idx] = np.fromfile(self.fd, count=self.nbead * 3, 
                                          dtype=self.dtype).reshape(
                                            (self.nbead, 3))
            self.cq.insert(0, idx)
            return self.structs[idx] 

    def read_bead(self, idx):
        v = self.beads[idx]
        for i in range(self.nstruct):
            self.fd.seek(self.headersize + self.structsize * i + 
                        self.crd3size * idx)
            v[i] = np.fromfile(self.fd, count=3, dtype=self.dtype)

    def get_bead(self, idx):
        idx = int(idx)
        if idx in self.beads:
            pos = self.bq.index(idx)
            self.bq.insert(0, self.bq.pop(pos))
            return self.beads[idx]
        else:
            self.free_space(self.beadsize)
            self.beads[idx] = np.empty((self.nstruct, 3))
            self.read_bead(idx)
            self.bq.insert(0, idx)
            return self.beads[idx]

    def _write_header(self):
        self.fd.seek(0)
        np.int32(len(self.shape)).tofile(self.fd)
        for dim in self.shape:
            np.int32(dim).tofile(self.fd)

    def set_struct(self, idx, struct):
        idx = int(idx)
        assert struct.shape == (self.nbead, 3)
        if idx in self.structs:
            pos = self.cq.index(idx)
            self.cq.insert(0, self.cq.pop(pos))    
        else:
            self.free_space(self.structsize)
            self.cq.insert(0, idx)
        
        self._cwrt[idx] = True
        self.structs[idx] = struct.astype(self.dtype, 'C', 'same_kind')

    def set_bead(self, idx, bead):
        idx = int(idx)
        assert bead.shape == (self.nstruct, 3)
        if idx in self.beads:
            pos = self.bq.index(idx)
            self.bq.insert(0, self.bq.pop(pos))    
        else:
            self.free_space(self.beadsize)
            self.bq.insert(0, idx)
        
        self._bwrt[idx] = True
        self.beads[idx] = bead.astype(self.dtype, 'C', 'same_kind')

    def get_all(self): # ignores limits
        self.flush()
        self.fd.seek(self.headersize)
        return np.fromfile(self.fd, count=self.nbead * self.nstruct * 3, 
                           dtype=self.dtype).reshape(
                           self.shape)

    def flush(self):
        for c in self._cwrt:
            self.dump_coord(c)
        for b in self._bwrt:
            self.dump_bead(b)

    def close(self):
        try:
            self.flush()
            self.fd.close()
        except:
            pass

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            self.flush()
            self.fd.close()
        except:
            pass

def transpose_popcrd_ondisk(fin, fout, max_memory='3GB', dtype='float32'):
    max_memory = parseSize(max_memory)
    pin = PopulationCrdFile(fin, 'r', dtype='float32', max_memory=max_memory//2)
    newshape = np.array([pin.shape[1], pin.shape[0], 3])
    pout = PopulationCrdFile(fout, 'w', dtype='float32', shape=newshape, 
                             max_memory=max_memory//2)
    for i in range(pin.nbead):
        pout.set_struct(i, pin.get_bead(i))


def transpose_popcrd_onmemory(fin, fout, max_memory='3GB', dtype='float32'):
    max_memory = parseSize(max_memory)
    pin = PopulationCrdFile(fin, 'r', dtype='float32', max_memory=max_memory//2)
    newshape = np.array([pin.shape[1], pin.shape[0], 3])
    pout = PopulationCrdFile(fout, 'w', dtype='float32', shape=newshape, 
                             max_memory=max_memory//2)
    #for i in range(pin.nbead):
        #pout.set_struct(i, pin.get_bead(i))
    crd = pin.get_all()
    crd = np.asarray(np.swapaxes(crd, 0, 1), order ="C", dtype=np.float32)
    pout.fd.seek(pout.headersize)
    crd.tofile(pout.fd)


class TransposedMatAcc(object):
    '''
    This class provides an interface to a transposed binary crd file created 
    by transpose_popcrd_* functions, so that slice access as [:, [1, 2, 3], :] 
    will return all the coordinates for beads [1, 2, 3].
    Transposing the file makes access a lot faster during HiC and SPRITE 
    calculations.
    '''
    def __init__(self, fname):
        self.p = PopulationCrdFile(fname)
        s, b, _ = self.p.shape
        self.shape = np.array([b, s, 3])
    def __getitem__(self, key):
        if isinstance(key, tuple):
            structs, beads, xyz  = key
            if isinstance(beads, int):
                return self.p.get_struct(beads)
            elif isinstance(beads, slice):
                raise NotImplementedError(str(type(beads)))
            else:
                #assume iterable
                try:
                    return np.swapaxes(
                        np.array([self.p.get_struct(e) for e in beads]),
                        0, 1)
                except TypeError:
                    raise NotImplementedError(str(type(beads)))