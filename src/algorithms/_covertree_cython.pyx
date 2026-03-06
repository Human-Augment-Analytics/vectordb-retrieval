# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrtf


ctypedef cnp.float32_t float32_t


def l2_distances_to_query(
    cnp.ndarray[float32_t, ndim=1] query,
    cnp.ndarray[float32_t, ndim=2] vectors,
):
    cdef Py_ssize_t n = vectors.shape[0]
    cdef Py_ssize_t d = vectors.shape[1]
    cdef Py_ssize_t i, j
    cdef float32_t diff
    cdef float32_t accum
    cdef cnp.ndarray[float32_t, ndim=1] out = np.empty(n, dtype=np.float32)

    for i in range(n):
        accum = 0.0
        for j in range(d):
            diff = vectors[i, j] - query[j]
            accum += diff * diff
        out[i] = sqrtf(accum)
    return out


def cosine_distances_to_query(
    cnp.ndarray[float32_t, ndim=1] query,
    cnp.ndarray[float32_t, ndim=2] vectors,
):
    cdef Py_ssize_t n = vectors.shape[0]
    cdef Py_ssize_t d = vectors.shape[1]
    cdef Py_ssize_t i, j
    cdef float32_t dot
    cdef cnp.ndarray[float32_t, ndim=1] out = np.empty(n, dtype=np.float32)

    for i in range(n):
        dot = 0.0
        for j in range(d):
            dot += vectors[i, j] * query[j]
        out[i] = 1.0 - dot
    return out


def negative_dot_to_query(
    cnp.ndarray[float32_t, ndim=1] query,
    cnp.ndarray[float32_t, ndim=2] vectors,
):
    cdef Py_ssize_t n = vectors.shape[0]
    cdef Py_ssize_t d = vectors.shape[1]
    cdef Py_ssize_t i, j
    cdef float32_t dot
    cdef cnp.ndarray[float32_t, ndim=1] out = np.empty(n, dtype=np.float32)

    for i in range(n):
        dot = 0.0
        for j in range(d):
            dot += vectors[i, j] * query[j]
        out[i] = -dot
    return out
