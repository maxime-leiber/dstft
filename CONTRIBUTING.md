# Contributing to dstft

Thanks for your interest in contributing! This document covers how to set up
a development environment, run the test suite and pre-commit checks, and
submit a pull request.

## Development setup

Clone the repository and install it in editable mode with the development
extras:

```bash
git clone https://github.com/maxime-leiber/dstft.git
cd dstft
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

See the [Installation](README.md#installation) section of the README for
Conda/Mamba + `uv` alternatives.

To also build the documentation locally, install the `docs` extra as well
(`pip install -e ".[dev,docs]"` installs both at once):

```bash
pip install -e ".[docs]"
```

## Running the test suite

```bash
python -m pytest --cov=dstft --cov-report=term-missing
```

CI runs this against Python 3.10, 3.11, and 3.12 — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Pre-commit checks

This repo enforces formatting, linting, `mypy`, and docstring coverage via
[pre-commit](https://pre-commit.com) (config in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml)). Install the hooks once
so they run automatically on `git commit`:

```bash
pre-commit install
```

Or run them on demand against the whole repo — this is what the `lint` job
in CI runs:

```bash
pre-commit run --all-files
```

## Commit messages

Commit subject lines are written in the imperative mood and describe what
the commit does, e.g. `Fix hop_length getting no/incomplete gradient with a
scalar window_mode`. There is no enforced prefix convention (no
`feat:`/`fix:`); use the commit body to explain *why* when the reason isn't
obvious from the diff or from the code itself.

## Submitting a pull request

1. Fork the repository and create a branch for your change.
2. Make your change, with tests covering new behavior and/or bug fixes.
3. Run `pytest` and `pre-commit run --all-files` locally; both must pass
   (CI enforces the same checks).
4. Open a pull request against `main` describing what changed and why —
   see recently merged PRs for the expected level of detail.

Changes to `src/dstft/` that alter numerical behavior (new formulas, new
modes, changed defaults) get extra scrutiny: please explain the motivation
and, where relevant, include a comparison against the previous behavior
(e.g. a gradcheck or a `torch.stft` parity check).

## Reporting bugs or requesting features

Please [open an issue](https://github.com/maxime-leiber/dstft/issues) using
the issue templates, which list the information that's most useful to
include.
