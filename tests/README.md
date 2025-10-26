# GWSNR Test Suite

Comprehensive test suite for validating GWSNR functionality across unit and integration scenarios. The test suite ensures reliability, accuracy, and performance of gravitational wave signal-to-noise ratio calculations.

## Test Organization

### Unit Tests (`unit/`)
Component-level tests validating individual GWSNR methods and backends:
- **SNR Calculation Methods**: Interpolation, inner product, and ANN-based approaches
- **Backend Implementations**: Numba, JAX, and MLX acceleration frameworks
- **Detection Statistics**: Probability of detection (Pdet) and threshold optimization
- **Performance Validation**: Cross-validation between different computational backends

### Integration Tests (`integration/`)
End-to-end workflow tests for astrophysical applications:
- **Horizon Distance Calculations**: BBH detection range estimates
- **Population Studies**: Detectable fraction and selection effects
- **Rate Calculations**: BBH merger rate estimations using detection probabilities

## Running Tests

### All Tests
```bash
pytest tests/ -v -s
```

### Unit Tests Only
```bash
pytest tests/unit/ -v -s
```

### Integration Tests Only
```bash
pytest tests/integration/ -v -s
```

### Individual Test Files
```bash
pytest tests/unit/test_GWSNR_interpolation_default_numba.py -v -s
pytest tests/integration/test_bbh_horizon_distance.py -v -s
```

## Test Directory Structure

```
tests/
├── README.md                              # This file
├── unit/                                  # Unit tests
│   ├── README.md                         # Unit test documentation
│   ├── unit_utils.py                     # Common test utilities and fixtures
│   ├── injection_data.json               # Test data for unit tests
│   ├── test_GWSNR_interpolation_default_numba.py
│   ├── test_GWSNR_interpolation_jax.py
│   ├── test_GWSNR_interpolation_mlx.py
│   ├── test_GWSNR_inner_product.py
│   ├── test_GWSNR_inner_product_jax.py
│   ├── test_GWSNR_ann.py
│   ├── test_GWSNR_pdet.py
│   ├── test_GWSNR_snr_recalculation.py
│   └── test_GWSNR_threshold.py
└── integration/                           # Integration tests
    ├── README.md                         # Integration test documentation
    ├── bbh_gw_params.json               # Astrophysical BBH parameters
    ├── test_bbh_horizon_distance.py
    ├── test_bbh_population_detectable_fraction.py
    └── test_bbh_rate.py
```

## Test Coverage

- **Computational Backends**: Numba, JAX, MLX implementations
- **SNR Methods**: Interpolation, inner product, neural networks
- **Detection Statistics**: Boolean and probabilistic detection methods
- **Astrophysical Applications**: Horizon distances, rates, selection effects
- **Performance**: Timing, reproducibility, cross-validation
- **Error Handling**: Invalid inputs, edge cases, exception handling

## Requirements

- **Core**: `pytest`, `numpy`, `gwsnr`
- **Optional**: `jax` (JAX tests), `mlx` (MLX tests), `ripplegw` (inner_product_jax tests), `tensorflow` and `scikit-learn` (ANN tests). Upgrading `ml-dtypes` may be required for compatibility.
- **Data**: Test data files (`injection_data.json`, `bbh_gw_params.json`) included

## Performance Expectations

- **Full Suite**: Complete test suite runs in within 5 minutes. 