# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def bs_eval(DTYPE_t r):
    cdef DTYPE_t retval = 0.0
    cdef DTYPE_t d = fabs(r)

    if d >= 2.0:
        return retval

    if d < 1:
        retval = (4.0 - 6.0 * d * d + 3.0 * d * d * d) / 6.0
        return retval

    retval = (2.0 - d)
    retval = retval * retval * retval / 6.0
    return retval


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _bs_eval_dist(DTYPE_t d):
    cdef DTYPE_t retval = 0.0

    if d >= 2.0:
        return retval

    if d < 1:
        retval = (4.0 - 6.0 * d * d + 3.0 * d * d * d) / 6.0
        return retval

    retval = (2.0 - d)
    retval = retval * retval * retval / 6.0
    return retval


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def tp_cubic(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] t):
    cdef DTYPE_t retval = 0.0
    cdef DTYPE_t d0 = fabs(x[0] - t[0])
    if d0 >= 2.0:
        return retval

    cdef DTYPE_t d1 = fabs(x[1] - t[1])
    if d1 >= 2.0:
        return retval

    cdef DTYPE_t d2 = fabs(x[2] - t[2])
    if d2 >= 2.0:
        return retval

    retval = _bs_eval_dist(d0) * _bs_eval_dist(d1) * _bs_eval_dist(d2)
    return retval

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def design_matrix(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] t):
    cdef int n = x.shape[0]
    cdef np.ndarray retval = np.zeros([n], dtype=DTYPE)

    for i in range(n):
        retval[i] = tp_cubic(x[i], t)

    return retval

