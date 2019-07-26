from distutils.core import setup
from Cython.Build import cythonize
import os


setup(
    ext_modules = cythonize("dynamic_window.pyx", annotate=True),
    name="lib_dwa",
)

