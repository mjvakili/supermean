from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import os
import numpy

ext_modules = []

ext_modules.append(Extension("terminator",
                             ["terminator.pyx"],
                             libraries=["m"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             )
                   )

setup(include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules, 
      )
"""

from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy as np
setup(
    name = 'terminato',
    ext_modules = cythonize(["terminator.pyx"]),
    include_dirs=[np.get_include()]
)
"""
