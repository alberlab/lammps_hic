from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
      name = 'sprite_assignment',
      ext_modules = cythonize(
          'sprite_assignment.pyx'
      ),
      include_dirs=[numpy.get_include()],
)
