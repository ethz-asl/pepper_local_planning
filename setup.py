import numpy
from setuptools import setup
from Cython.Build import cythonize

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="asl-pepper-responsive",
    description='libraries for DWA low-latency planner for pepper',
    author='Daniel Dugas',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ethz-asl/pepper_local_planning.git",
    version='0.0.1',
    py_modules=[],
    install_requires=['matplotlib'],
    ext_modules= cythonize(["lib_dwa/dynamic_window.pyx", "lib_clustering/clustering.pyx"], annotate=True),
    include_dirs=[numpy.get_include()],
)

