# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog starts from `v3.0.0` (the first version published on PyPI);
earlier history isn't backfilled.

## [Unreleased]

### Added

- Standard PyPI installation instructions (`pip install dstft`, `uv pip
  install dstft`, Conda/Mamba) in `README.md` and the docs, alongside the
  existing editable/dev install.
- `CITATION.cff` and a landing-page pitch paragraph in `README.md`/docs.
- Colab badge and an expanded usage example in `README.md`.
- Tests: gradient-check (`gradcheck`), `torch.stft` parity, and CUDA
  device tests; CI now also runs on Python 3.12.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- `.github/pull_request_template.md`, `.github/ISSUE_TEMPLATE/bug_report.md`,
  `.github/ISSUE_TEMPLATE/feature_request.md`.
- `.gitattributes` for consistent line endings.

### Changed

- Test coverage raised from 86% to 97% (`_core.py`, `visualization.py`,
  and `windows.py` are now at 100%).

### Removed

- `overlap_add_wola`: dead code superseded by `overlap_add_dual`, not
  part of the public API.

[Unreleased]: https://github.com/maxime-leiber/dstft/compare/v3.0.0...HEAD
