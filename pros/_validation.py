"""Shared validation for public numerical entry points."""

import operator

import numpy as np


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be within the valid range")
    return value


def matrix(X, name="X"):
    raw = np.asarray(X)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    X = np.ascontiguousarray(raw, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError(f"{name} must be a non-empty 2-D coordinate matrix")
    # Chunk the finite check rather than allocate an N*d boolean temporary.
    for start in range(0, len(X), 4096):
        if not np.isfinite(X[start : start + 4096]).all():
            raise ValueError(f"{name} must contain only finite values")
    return X


def indices(values, size, name="indices", allow_empty=False):
    a = np.asarray(values)
    if a.ndim != 1 or (not allow_empty and not a.size):
        raise ValueError(f"{name} must be a non-empty 1-D integer array")
    if a.size and (a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= size)):
        raise ValueError(f"{name} must contain valid indices (integer rows)")
    a = a.astype(np.int64)
    if np.unique(a).size != a.size:
        raise ValueError(f"{name} must not contain duplicates")
    return a
