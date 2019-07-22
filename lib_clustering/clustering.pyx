# distutils: language=c++

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
cimport cython
from math import sqrt
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport acos as cacos
from libc.math cimport sqrt as csqrt

import os
from yaml import load
from matplotlib.pyplot import imread

def ranges_to_xy(np.float32_t[:] ranges, np.float32_t[:] angles,
                 np.float32_t[:] x, np.float32_t[:] y):
    for i in range(ranges.shape[0]):
        x[i] = ranges[i] * ccos(angles[i])
        y[i] = ranges[i] * csin(angles[i])



def euclidean_clustering(np.float32_t[:] ranges):
    # assume angles from 0 to 2pi
    cdef np.float32_t[:] angles = np.linspace(0, 2*np.pi, ranges.shape[0]+1)[:-1]
    cdef np.float32_t[:] x = np.zeros((n_points), dtype=np.float32)
    cdef np.float32_t[:] y = np.zeros((n_points), dtype=np.float32)
    ranges_to_xy(ranges, angles, x, y)

    THRESH = 0.1
    THRESH_SQ = THRESH * THRESH

    clusters = []
    cdef bool create_new_cluster = True
    for i in range(x.shape[0]):
        create_new_cluster = True
        xi = x[i]
        yi = y[i]
        for c in clusters:
            for p in c:
                dx = xi - x[p]
                dy = yi - y[p]
                if ( dx * dx + dy * dy ) < THRESH_SQ:
                    c.append(p)
                    create_new_cluster = False
        if create_new_cluster:
            clusters.append([i])

    return clusters

