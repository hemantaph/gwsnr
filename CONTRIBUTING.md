# Community Guidelines

Thank you for your interest in `gwsnr`. Contributions, bug reports, questions,
examples, and performance improvements are welcome. These guidelines explain how
to contribute, report problems, and seek support.

## Ways to contribute

Useful contributions include:

- bug fixes
- documentation improvements
- examples and tutorials
- tests
- new SNR, waveform, interpolation, or detection-probability features
- performance improvements for NumPy, Numba, JAX, MLX, or multiprocessing paths

For larger changes, please open an issue first so we can discuss the scope,
expected behavior, and scientific assumptions before implementation work grows
too large.

## Development setup

1. Fork the repository and clone your fork.
2. Create a Python environment with Python 3.10 or newer.
3. Install `gwsnr` in editable mode with development tools:

```bash
pip install -e ".[dev]"
```

4. Install optional backend dependencies only when you need them:

```bash
pip install -e ".[jax]"
pip install -e ".[mlx]"
pip install -e ".[tensorflow]"
```

5. Create a branch for your work:

```bash
git checkout -b my-change
```

## Making changes

Please keep pull requests focused and easy to review. A good pull request:

- describes the problem and the proposed solution
- adds or updates tests when behavior changes
- updates documentation or examples when user-facing behavior changes
- explains numerical assumptions, approximations, or backend-specific behavior
- keeps unrelated formatting or refactoring out of the change

`gwsnr` has import-sensitive and performance-sensitive code paths. Avoid adding
heavy imports to package `__init__.py` files unless they are truly required, and
prefer lazy imports for optional dependencies and backend-specific modules.

Run the relevant test suite before opening a pull request:

```bash
python -m pytest tests/unit --tb=short
```

For changes that may affect end-to-end numerical behavior, also run:

```bash
python -m pytest tests/integration --tb=short
```

For backend-specific changes, run the matching tests when the dependencies are
installed, for example:

```bash
python -m pytest tests/unit/test_GWSNR_interpolation_jax.py --tb=short
python -m pytest tests/unit/_test_GWSNR_interpolation_mlx.py --tb=short
```

## Reporting issues or problems

Please report bugs through the GitHub issue tracker:

https://github.com/hemantaph/gwsnr/issues

When possible, include:

- the `gwsnr` version
- your Python version and operating system
- the command or code that produced the problem
- the full error message or traceback
- a small reproducible example
- the SNR method, waveform approximant, detectors, PSDs, and backend used
- any relevant input files, configuration, random seeds, or generated
  interpolators

If the issue is about a scientific result, please also describe the physical
model, assumptions, expected behavior, and any reference calculation used for
comparison.

## Seeking support

For questions about installation, usage, examples, or unexpected behavior,
please use the GitHub issue tracker:

https://github.com/hemantaph/gwsnr/issues

Before opening a new issue, please check:

- the documentation: https://gwsnr.hemantaph.com/
- existing issues: https://github.com/hemantaph/gwsnr/issues
- the examples in the `docs/examples/` directory

## Code of conduct

Please be respectful and constructive in all project discussions. We welcome
contributions from people with different backgrounds, experience levels, and
research interests. Harassment, personal attacks, and other exclusionary
behavior are not acceptable.

## Citation

If `gwsnr` supports your research, please cite the project as described in the
documentation and repository metadata.
