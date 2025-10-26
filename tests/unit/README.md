# Unit Tests

Component-level tests validating individual GWSNR methods, computational backends, and core functionality. Each test focuses on specific features and validates correctness, reproducibility, and performance.

## Test Files Overview

### SNR Calculation Methods

#### `test_GWSNR_interpolation_default_numba.py`
**Primary interpolation-based SNR calculations with Numba acceleration**
- Aligned spin BBH systems using IMRPhenomD approximant
- Output validation: structure, data types, shapes, numerical properties
- Computational reproducibility and deterministic results
- Error handling for invalid inputs (negative masses, NaN/Inf values)
- Custom configurations (mass ranges, waveform approximants, detector setups)
- JSON file output generation and integrity

#### `test_GWSNR_interpolation_jax.py`
**JAX-accelerated interpolation methods with JIT compilation**
- Cross-validation with standard interpolation methods
- JIT compilation performance optimization
- Multiple waveform approximants (IMRPhenomD, IMRPhenomXAS)
- Computational reproducibility with JAX backend
- GPU acceleration capabilities (when available)

#### `test_GWSNR_interpolation_mlx.py`
**MLX framework integration for Apple Silicon optimization**
- MLX-accelerated interpolation calculations
- Apple Silicon-specific optimizations
- Cross-validation with standard methods
- Performance benchmarking on Apple hardware
- Compatibility testing for MLX-enabled systems

#### `test_GWSNR_inner_product.py`
**Direct inner product SNR calculations**
- Spin-precessing BBH systems using IMRPhenomXPHM approximant
- Multiple waveform approximants (IMRPhenomD, TaylorF2)
- Custom detector configurations and PSDs
- Serial vs parallel multiprocessing performance
- Cross-validation with interpolation methods

#### `test_GWSNR_inner_product_jax.py`
**JAX-accelerated inner product calculations using ripplegw**
- JAX/ripplegw vs LAL implementation comparison
- Multiple waveform approximants (IMRPhenomXAS, IMRPhenomD, IMRPhenomD_NRTidalv2)
- Cross-validation with standard inner product methods
- JIT compilation and performance optimization
- GPU acceleration support

#### `test_GWSNR_ann.py`
**Artificial Neural Network-based SNR predictions**
- ANN model validation for spinning BBH systems
- Cross-validation with direct inner product methods
- Performance: ANN vs full bilby recalculation
- Probability of detection consistency
- Spin-precessing systems with IMRPhenomXPHM

### Detection Statistics

#### `test_GWSNR_pdet.py`
**Probability of detection (Pdet) calculations**
- Boolean and probability distribution outputs
- Non-central chi-squared and Gaussian statistical distributions
- Output validation for different distribution types
- Consistent behavior across statistical models
- BBH systems with aligned spins

#### `test_GWSNR_threshold.py`
**SNR threshold optimization using cross-entropy methods**
- Cross-entropy optimization for threshold determination
- Real astrophysical BBH injection data (O4 catalog)
- Convergence and numerical stability validation
- Performance benchmarking for optimization algorithms
- Integration with injection parameters (mass, redshift, FAR, SNR)

#### `test_GWSNR_snr_recalculation.py`
**Hybrid SNR recalculation workflows**
- Selective recalculation based on SNR thresholds
- Interpolation + inner product hybrid approaches
- Performance optimization for large datasets
- Accuracy improvement validation
- Cross-validation between methods

## Shared Resources

### `unit_utils.py`
**Common test utilities and fixtures**
- `CommonTestUtils` base class with shared methods
- Parameter generation for BBH/BNS systems
- Output validation helpers
- Cross-validation utilities
- Performance timing helpers

### `injection_data.json`
**Test data for unit tests**
- 20,000 BBH injection parameters
- Reed Essick O4 injection catalog subset
- Contains: masses, spins, distances, sky locations, detection statistics
- Used for realistic parameter validation

## Running Unit Tests

### All Unit Tests
```bash
pytest tests/unit/ -v -s
```

### Individual Test Suites
```bash
# Interpolation methods
pytest tests/unit/test_GWSNR_interpolation_default_numba.py -v -s
pytest tests/unit/test_GWSNR_interpolation_jax.py -v -s
pytest tests/unit/test_GWSNR_interpolation_mlx.py -v -s

# Inner product methods
pytest tests/unit/test_GWSNR_inner_product.py -v -s
pytest tests/unit/test_GWSNR_inner_product_jax.py -v -s

# Detection statistics
pytest tests/unit/test_GWSNR_pdet.py -v -s
pytest tests/unit/test_GWSNR_threshold.py -v -s

# Neural networks and hybrid methods
pytest tests/unit/test_GWSNR_ann.py -v -s
pytest tests/unit/test_GWSNR_snr_recalculation.py -v -s
```

### Specific Test Methods
```bash
pytest tests/unit/test_GWSNR_interpolation_default_numba.py::TestGWSNRInterpolation::test_output_aligned_spins -v -s
```

## Test Categories

### **Core Functionality**
- Basic SNR calculations
- Output format validation
- Numerical accuracy checks
- Reproducibility validation

### **Computational Backends**
- Numba JIT compilation
- JAX acceleration and GPU support
- MLX Apple Silicon optimization
- Cross-validation between backends

### **Waveform Approximants**
- IMRPhenomD (aligned spins)
- IMRPhenomXPHM (precessing spins)
- TaylorF2 (post-Newtonian)
- IMRPhenomD_NRTidalv2 (neutron stars)

### **System Types**
- Binary Black Holes (BBH)
- Binary Neutron Stars (BNS)
- Aligned and precessing spins
- Mass ratio variations

### **Performance Validation**
- Timing benchmarks
- Memory usage
- Multiprocessing efficiency
- Cross-method comparison

## Expected Performance

- **Full Test Suite**: within 5 minutes for all unit tests

<!-- - **Individual Tests**: 10-60 seconds per test method

- **Memory Usage**: < 2GB for typical test runs
- **CPU Utilization**: Efficient multiprocessing when enabled  -->

## Dependencies

### Required

- `pytest`, `gwsnr`, `jax`, `jaxlib`, `ripplegw`, `scikit-learn`, `tensorflow`, `ml-dtypes`
- Test data files (`injection_data.json`)
- Optional: `mlx` for MLX tests with Apple Silicon Hardware. `"jax[cuda12]"` for Nvidia GPU support.


## Common Test Patterns

All unit tests follow consistent patterns:
- Configuration dictionaries for reproducible setups
- Inheritance from `CommonTestUtils` for shared functionality
- Standardized output validation methods
- Cross-validation between different computational approaches
- Performance timing and benchmarking