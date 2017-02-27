import sqlite3
import numpy as np
import json
import io
import os
import os.path
import h5py

_N_RETRY = 5


bond_type_tag = {'harmonic_upper_bound': 0,
                 'harmonic_lower_bound': 1}

bond_type_name = {i: name for name, i in bond_type_tag.items()}


def _adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("array", _convert_array)


def _create_struct_db(dbfile, n_struct, radii, chrom, description):
    with sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cur = conn.cursor()
        cur.execute('CREATE TABLE meta (nstruct int, radii array, chrom array, description text)')
        cur.execute('INSERT into meta VALUES (?, ?, ?, ?)', (n_struct,
                                                             np.array(radii),
                                                             np.array(chrom),
                                                             description))
        conn.commit()


def _get_meta(dbfile):
    with sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM meta')
        meta = cur.fetchone()
    return meta


class DBStructFile(object):
    """
    Interface to use a sqlite database to write coordinates.

    Arguments:
        path (str): path of the database file.
        mode (str): one of 'a', 'r' or 'w'. Defaults to 'a'.
            'a' creates the file if does not exist or works on the existing
            database if it does.
            'r' raises an exception if the file does not exist
            'w' remove the previous database if the path exists.
        n_struct (int): optional, number of structures
        radii (np.ndarray): optional, radius of the beads
        chrom (np.ndarray): optional, chromosome tags
        description (str): optional, any description string
        n_retry (int): number of retries if got an OperationalError
            while writing/reading

    Attributes:
        n_struct (int): number of structures
        radii (np.ndarray): radius of the beads
        chrom (np.ndarray): chromosome tags
        description (str): any description string
        n_retry (int): number of retries if got an OperationalError
            while writing/reading
    """
    def __init__(self, path, mode='a', n_struct=None, radii=None, chrom=None, description='', n_retry=_N_RETRY):

        self.dbfile = path
        self.N_RETRY = n_retry
        
        if mode == 'r':
            if not os.path.isfile(path):
                raise IOError('%s: No such file' % path)
            self.n_struct, self.radii, self.chrom, self.description = _get_meta(path)

        elif mode == 'w':
            if os.path.isfile(path):
                os.remove(path)
            _create_struct_db(path, n_struct, radii, chrom, description)

        elif mode == 'a':
            if os.path.isfile(path):
                self.n_struct, self.radii, self.chrom, self.description = _get_meta(path)
            else:
                _create_struct_db(path, n_struct, radii, chrom, description)

        else:
            raise ValueError('Invalid mode')


    def add_group(self, iter_name):
        '''
        Create a table to store coordinates, infos, and violations.

        The table name is set to *iter_name*, so be careful not to
        SQL inject yourself.

        Arguments:
            iter_name (str): Name of the iteration, es: 'p0.010a', 'p0.010b'.
                Avoid SQL injections, the function does not check for you.

        Returns:
            None
        '''
        with sqlite3.connect(self.dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cur = conn.cursor()
            cur.execute('CREATE TABLE "%s" (struct int PRIMARY KEY, crd array, info text, violations array)' % iter_name)
            conn.commit()


    def write_structure(self, iter_name, struct_num, crd, info={}, violations=[]):
        '''
        Writes a structure to database.
        
        Arguments:
            iter_name (str): Name of the iteration, es: 'p0.010a', 'p0.010b'
            struct_num (int): Number of the structure
            crd (numpy.ndarray): Structure coordinates
            info (dict): Information about the run
            violations (list): List of violations

        Returns:
            None

        Examples:
            f = DBStructFile('existing.db')
            crd = np.random.random((10,3))
            f.write_structure('p0.010a', 0, crd)
        '''
        # use json for info 
        cinfo = json.dumps(info)

        # use numpy array for violations, use a number to serialize bond types
        cviol = np.array([(i, j, absv, relv, bond_type_tag[bt]) for (i, j, absv, relv, bt) in violations], dtype='f4')
        n_try = 0
        while n_try < self.N_RETRY:
            try:
                with sqlite3.connect(self.dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                    cur = conn.cursor()
                    cur.execute('INSERT INTO "%s" VALUES (?, ?, ?, ?)' % iter_name, 
                                (struct_num, crd, cinfo, cviol))
                    conn.commit()
                    n_try = self.N_RETRY  # equivalent to break
            except sqlite3.OperationalError:
                n_try += 1
                if n_try >= _N_RETRY:
                    raise  


    def get_structure(self, iter_name, struct_num):
        '''
        Retrieve a single structure from the file.

        Arguments:
            iter_name (str): Name of the iteration, es: 'p0.010a', 'p0.010b'
            struct_num (int): Number of the structure

        Returns:
            A list containing coordinates, info and violations.

            (crd (numpy.ndarray), info (dict), violations (list))
        '''
        n_try = 0
        while n_try < self.N_RETRY:
            try:
                with sqlite3.connect(self.dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                    cur = conn.cursor()
                    cur.execute('SELECT crd, info, violations FROM'
                                ' "%s" WHERE struct=? LIMIT 1' % iter_name,
                                (struct_num,))
                    rec = cur.fetchone()
                    if rec is None:
                        return None
                    crd = rec[0]
                    cinfo = json.loads(rec[1])
                    cviol = [(int(i), int(j), absv, relv, bond_type_name(bt))
                             for (i, j, absv, relv, bt) in rec[2]]
                    return crd, cinfo, cviol
                    
            except sqlite3.OperationalError:
                n_try += 1
                if n_try >= _N_RETRY:
                    raise


    def to_hss(self, iter_name, path):
        '''
        Create an hss file from a table (iteration).

        Arguments:
            iter_name (str): Name of the iteration, es: 'p0.010a', 'p0.010b'
            path (str): Output filename

        Returns:
            None

        Raises:
            IOError if the number of records do not correspond to n_struct
        '''
        with sqlite3.connect(self.dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM "%s"' % iter_name)
            cnt = cur.fetchone()[0]
            if cnt != self.n_struct:
                raise IOError('Incorrect number of records to create hss. Unfinished run?')
            
            cur.execute('SELECT struct, crd FROM "%s"' % iter_name)
            res = cur.fetchone()

            i, crd = res
            n_beads = len(crd)
            
            with h5py.File(path, 'w') as f: 
                f.create_dataset('coordinates', shape=(self.n_struct, n_beads, 3), dtype='f4')
                f.create_dataset('radius', data=self.radii, dtype='f4')
                f.create_dataset('idx', data=self.chrom)
                f.create_dataset('nstruct', data=self.n_struct, dtype='i4')
                f.create_dataset('nbead', data=n_beads, dtype='i4')
                f['coordinates'][i] = crd
                
                res = cur.fetchone()
                while res is not None:
                    i, crd = res
                    f['coordinates'][i] = crd
                    res = cur.fetchone()


    def get_coordinates(self, iter_name):
        with sqlite3.connect(self.dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cur = conn.cursor()
            cur.execute('SELECT struct, crd FROM "%s"' % iter_name)
            res = cur.fetchone()
            if res is None:
                raise IOError('No structures for %s' % iter_name)

            i, crd = res
            coords = np.empty((self.n_struct, len(crd), 3))
            coords[i] = crd

            res = cur.fetchmany()
            while len(res) > 0:
                for i, crd in res:
                    coords[i] = crd
                res = cur.fetchmany()

            return coords


    def sqldump(self):
        """
        Returns the sql dump of the database.
        """
        with sqlite3.connect(self.dbfile, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            dmp = '\n'.join([line for line in conn.iterdump()])
        return dmp



