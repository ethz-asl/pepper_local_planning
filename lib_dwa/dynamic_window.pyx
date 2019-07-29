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




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def linear_dwa(np.float32_t[:] s_next,
        np.float32_t[:] angles,
        u, v, gx, gy, dt,
        DV=0.1, # velocity sampling resolution
        UMIN=-0.5,
        UMAX=0.5,
        VMIN=-0.5,
        VMAX=0.5,
        AMAX=0.4,
        COMFORT_RADIUS_M=0.5,
        PLANNING_RADIUS_M=0.7,
        ):

    # Specification vel limits
    W = [UMIN, UMAX, VMIN, VMAX]




    # sample in window
    cdef np.float32_t best_score = -1000
    cdef np.float32_t best_u = 0
    cdef np.float32_t best_v = 0
    cdef np.float32_t du = 0
    cdef np.float32_t dv = 0
    cdef np.float32_t norm_duv = 0
    cdef np.float32_t max_norm_duv = AMAX * dt
    cdef np.float32_t max_norm_uv = VMAX
    cdef np.float32_t us = 0
    cdef np.float32_t vs = 0
    cdef np.float32_t xs = 0
    cdef np.float32_t ys = 0
    cdef np.float32_t dgxs = 0 # sampled x dist to goal
    cdef np.float32_t dgys = 0 # sampled y dist to goal
    cdef np.float32_t min_dist = 100
    cdef np.float32_t min_dist_s = 100
    cdef np.float32_t improvement = 0
    cdef np.float32_t scan_score = 0
    cdef np.float32_t goal_score = 0
    cdef np.float32_t score = 0
    cdef np.float32_t cgx = np.float32(gx)
    cdef np.float32_t cgy = np.float32(gy)
    cdef np.float32_t goal_norm = csqrt(gx * gx + gy * gy)
    cdef np.float32_t goal_norm_s = 0
    cdef np.float32_t cMAX_GOAL_NORM = 3
    cdef np.float32_t cdt = np.float32(dt)
    cdef np.float32_t cCOMFORT_RADIUS_M = np.float32(COMFORT_RADIUS_M)
    cdef np.float32_t cPLANNING_RADIUS_M = np.float32(PLANNING_RADIUS_M)
    cdef np.float32_t[:] us_list = np.arange(UMIN, UMAX, DV, dtype=np.float32)
    cdef np.float32_t[:] vs_list = np.arange(VMIN, VMAX, DV, dtype=np.float32)
    cdef np.float32_t[:] s_next_shift = np.zeros_like(s_next, dtype=np.float32)
    # find current closest point
    for k in range(len(s_next)):
        r = s_next[k] 
        if r == 0:
            continue
        if r < min_dist:
            min_dist = r
    # sample in window and score
    for i in range(len(us_list)):
        us = us_list[i]
        for j in range(len(vs_list)):
            vs = vs_list[j]
            # dynamic limits as condition (circle around current u v)
            du = us - u
            dv = vs - v
            norm_duv = csqrt(du * du + dv * dv)
            if norm_duv > max_norm_duv:
                continue
            # specification vel limits corner case (ellipsid around 0, 0)
            norm_uv = csqrt(us * us + vs * vs)
            if norm_uv > max_norm_uv:
                continue
            # motion model TODO refine
            xs = us * cdt
            ys = vs * cdt
            min_dist_s = 100
            # goal score is the diference between the current goal distance and the sampled one
            dgxs = cgx - xs
            dgys = cgy - ys
            goal_norm_s = csqrt(dgxs * dgxs + dgys * dgys)
            goal_score = goal_norm - goal_norm_s
            # refuse to exceed max goal dist
            if goal_norm > cMAX_GOAL_NORM:
                if goal_score < 0:
                    continue
            # scan score - shift the scan by dx, dy, then find the smallest range (closest point)
            for k in range(len(s_next)):
                # TODO potential optim. : precompute smaller scan with only ranges close enough to matter
                r = s_next[k] 
                if r == 0:
                    continue
                shifted_r = r - xs * ccos(angles[k]) - ys * csin(angles[k])
                s_next_shift[k] = shifted_r
                if shifted_r < min_dist_s:
                    min_dist_s = shifted_r
            if min_dist > cCOMFORT_RADIUS_M:
                # normal situation
                if min_dist_s < cCOMFORT_RADIUS_M:
                    scan_score = 0
#                 elif min_dist_s < cPLANNING_RADIUS_M: # linear ramp 0 to 1
#                     scan_score = (min_dist_s - cCOMFORT_RADIUS_M) / (cPLANNING_RADIUS_M - cCOMFORT_RADIUS_M)
                else:
                    scan_score = 1
            else:
                # we are far inside an obstacle, priority is evasion TODO: improve 
                improvement = min_dist_s - min_dist
                if improvement > 0:
                    scan_score = improvement + 0.1 * goal_score
                    goal_score = 1
                else:
                    scan_score = 0
            score = scan_score * goal_score
#             print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(gx, gy, xs, ys, goal_score, score))
            if score > best_score:
                best_score = score
                best_u = us
                best_v = vs

    return best_u, best_v, best_score



