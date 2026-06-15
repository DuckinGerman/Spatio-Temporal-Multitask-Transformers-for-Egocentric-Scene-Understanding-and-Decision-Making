# -*- coding: utf-8 -*-
"""Compat patch for NumPy>=1.24 where np.float/np.int/np.bool are removed."""
import numpy as np  # type: ignore

# Map removed aliases to proper dtypes to satisfy 3rd-party libs (e.g. pycocotools)
if not hasattr(np, 'float'):
    np.float = np.float64  # type: ignore[attr-defined]
if not hasattr(np, 'int'):
    np.int = np.int_  # type: ignore[attr-defined]
if not hasattr(np, 'bool'):
    np.bool = np.bool_  # type: ignore[attr-defined]
if not hasattr(np, 'object'):
    np.object = np.object_  # type: ignore[attr-defined]