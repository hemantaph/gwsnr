# gwsnr: Gravitational Wave Signal-to-Noise Ratio Computation Package
[![DOI](https://zenodo.org/badge/626733473.svg)]() [![PyPI version](https://badge.fury.io/py/ler.svg)](https://badge.fury.io/py/gwsnr) [![DOCS](https://readthedocs.org/projects/gwsnr/badge/?version=latest)](https://gwsnr.readthedocs.io/en/latest/)

## Summary

Gravitational waves are ripples in spacetime predicted by Einstein's theory of General Relativity. Detected for the first time in 2015, these waves, emanating from events like black hole or neutron star mergers, have opened new avenues in astrophysics. The Signal-to-Noise Ratio (SNR) is a critical measure in gravitational wave analysis, representing the signal strength relative to background noise in detectors like LIGO or Virgo. However, efficient computation of SNR, especially in simulations and hierarchical Bayesian analyses, is a complex and time-consuming task. The `gwsnr` Python package addresses this challenge by providing efficient tools for SNR computation.

## Statement of Need

The `gwsnr` package is designed to facilitate efficient and accurate SNR computations in gravitational wave research. It implements advanced techniques for enhancing calculation speed and precision, making it a valuable tool for researchers in this field. Key features include:

- An innovative half-scaling interpolation method for spin-less binary systems, focusing on mass parameters.
- A noise-weighted inner product method, similar to the one in `bilby`, enhanced with multiprocessing for parallel processing.
- Integration of the `numba` Just-In-Time (njit) compiler for optimized performance.
- User-friendly interface and compatibility with other gravitational wave analysis software.

The package is particularly useful in simulations of binary mergers, calculation of merger rates, gravitational wave lensing rates analysis, and in hierarchical Bayesian frameworks for analyzing selection effects. Additionally, it supports the `LeR` package for calculating detectable rates of lensed and unlensed gravitational wave events.

## Mathematical Formulation

### Modified FINDCHIRP Method: Half Scaling Approach

The `gwsnr` package introduces a Half Scaling method for SNR calculations, based on the FINDCHIRP algorithm. It focuses on non-spinning IMR waveforms and interpolates the Half scaled SNR based on mass parameters. Key aspects include:

- A 2D cubic spline interpolation method for the 'halfsnr' segment.
- The optimal SNR for a simple inspiral waveform is functionally dependent on various parameters and the detector's noise curve.
- The half scaled SNR is approximated and considered a function of total mass and mass ratio.

### Noise-Weighted Inner Product Method

This method is suited for SNR calculations in systems with frequency domain waveforms, including spin-precessing binary systems. It involves:

- Multi-process waveform generation, antenna pattern calculation, and noise-weighted inner product computation.
- The calculation of the optimal SNR involves integrating over frequency domain waveform polarizations and antenna patterns.

These methods underscore the `gwsnr` package's ability to handle a wide range of gravitational wave signals with enhanced efficiency and accuracy.

## Installation and Usage

Install `gwsnr` with `pip install gwsnr`

## Contributing

[Guidelines for contributing to the package.]

## License

[License details.]

## Contact

[Contact information for support or collaboration.]

## Acknowledgements

[Acknowledgements and credits.]

## References

1. [Link to the FINDCHIRP paper and other relevant publications.]

(Note: The above `readme.md` content provides a comprehensive overview of the `gwsnr` package, suitable for a GitHub repository. It outlines the package's purpose, key features, mathematical basis, and other essential information.)
