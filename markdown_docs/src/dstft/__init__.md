# `dstft/__init__.py`

The package's public entry point. This file's only job is to decide what
`import dstft` actually exposes — it contains no logic of its own.

## What it exports

```python
__all__ = ["DSTFT", "__version__", "plot_spec", "plot_win_lengths"]
```

- **`DSTFT`** — re-exported from `dstft.dstft`. This is the one class users
  are expected to instantiate; see `dstft.md`.
- **`plot_spec`, `plot_win_lengths`** — re-exported from `dstft.visualization`,
  so `dstft.plot_spec(...)` works without reaching into the submodule. The
  same two functions are also available as bound methods on a `DSTFT`
  instance (`DSTFT.plot_spec`, `DSTFT.plot_win_lengths`) for convenience.
- **`__version__`** — resolved at import time via
  `importlib.metadata.version("dstft")`, i.e. from the installed package's
  metadata rather than a hardcoded string in the source. This is what makes
  the version follow the git tag automatically (see `[tool.setuptools_scm]`
  in `pyproject.toml`) instead of needing to be bumped by hand in two places.
  If the package isn't installed (e.g. running from a raw source checkout
  without `pip install -e .`), resolution raises `PackageNotFoundError`,
  which is caught and falls back to the literal string `"0.0.0+unknown"`
  rather than crashing on import.

## Why this matters for the public API surface

Anything not listed in `__all__` (and not one of these four names) is
internal, even if it's technically importable with a longer dotted path —
`dstft._core`, `dstft.windows`, and the private `_`-prefixed attributes on
`DSTFT` are implementation details that can change without a semver-major
bump. If you find yourself importing from `dstft._core` directly, that's a
signal the public `DSTFT` API is missing something you need.
