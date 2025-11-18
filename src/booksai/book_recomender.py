
"""Compatibility shim for the misspelled module name `book_recomender`.

This file exists to avoid import errors if external code or older scripts
import the misspelled module name. It re-exports the main `recommender`
symbols and emits a gentle deprecation warning.
"""
from __future__ import annotations

import warnings

# Re-export the public symbols from the canonical recommender module
try:
	from .recommender import *  # noqa: F401,F403
	warnings.warn(
		"Importing `booksai.book_recomender` is deprecated; import `booksai.recommender` instead.",
		DeprecationWarning,
	)
except Exception:  # pragma: no cover - keep import-safe fallback
	# If the recommender package is not available (e.g., minimal installs),
	# keep this module importable but minimal.
	__all__ = []
