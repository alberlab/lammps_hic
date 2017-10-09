from distutils.core import setup
from Cython.Build import cythonize

setup(
      name = 'sprite_assignment',
      ext_modules = cythonize(
          'sprite_assignment.pyx'
      ),
)
