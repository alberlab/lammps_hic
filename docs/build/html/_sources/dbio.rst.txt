Database IO
===========

This module provides a wrapper class to write/read from a database instead 
of hdf5 files. Uses sqlite3, which is not fast, but should serve well if 
write request are not too frequent.

Note that a sqlite database will require ~2 times the disk space with
respect to a hdf5 file


DBStructFile class reference
----------------------------

.. autoclass:: lammps_hic.dbio.DBStructFile
   :members:
