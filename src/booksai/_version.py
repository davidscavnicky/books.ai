def _get_version() -> str:
    # we use lazy imports to avoid importing modules that are not
    # used when the use of this function is patched out at build time
    try:
        # prefer stdlib import (Python 3.9+)
        from importlib.resources import files  # type: ignore
    except Exception:
        # fall back to the backport package if present
        try:
            from importlib_resources import files  # type: ignore
        except Exception:
            files = None  # type: ignore

    from pathlib import Path  # noqa: PLC0415

    try:
        import versioningit  # noqa: PLC0415
    except Exception:
        # versioningit isn't available in this environment (dev/CI),
        # return a sensible default version string instead of failing import
        return "0.0"

    if files is None:
        return "0.0"

    module_path = files("booksai")
    if isinstance(module_path, Path):
        return versioningit.get_version(project_dir=Path(module_path).parent.parent)
    else:
        return "0.0"


__version__ = _get_version()
