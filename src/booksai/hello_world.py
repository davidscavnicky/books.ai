from __future__ import annotations

"""Utility helpers used by tests and small demos.

This module avoids executing network/download code at import time. Optional
dependencies like `kagglehub` are imported lazily inside functions so that
the module can be imported in environments where those packages are not
installed (e.g., CI that only runs unit tests).
"""

from typing import Any


def greeter(name: str) -> str:
	return f"Hello, {name}!"


def hello_world() -> None:
	"""Print a friendly hello world message."""
	print("Hello, world!")


def myadd(a: int, b: int) -> int:
	return int(a) + int(b)


def download_dataset(dataset: str = "arashnic/book-recommendation-dataset") -> Any:
	try:
		import kagglehub  # type: ignore
	except Exception as exc:  # pragma: no cover - env-dependent
		raise ImportError(
			"kagglehub is not installed in this environment; install it to use download_dataset"
		) from exc

	return kagglehub.dataset_download(dataset)