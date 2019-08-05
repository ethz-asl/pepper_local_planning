from distutils.core import setup
from Cython.Build import cythonize
import os


setup(
    ext_modules = cythonize("clustering.pyx", annotate=True),
    name="lib_clustering",
)

