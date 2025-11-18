"""Compatibility shim for the recommender implementation.

The original implementation was moved to `recommender_deprecated.py`.
This module re-exports the main functions and emits a DeprecationWarning
so external scripts keep working while code is migrated.
"""
from __future__ import annotations

import warnings

from .recommender_deprecated import (
    load_data,
    popularity_recommender,
    build_item_item_matrix,
    item_item_recommender,
    content_based_recommender,
)


warnings.warn(
    "booksai.recommender is deprecated; use recommender_deprecated or the new APIs instead",
    DeprecationWarning,
)

__all__ = [
    'load_data',
    'popularity_recommender',
    'build_item_item_matrix',
    'item_item_recommender',
    'content_based_recommender',
]
