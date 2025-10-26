# Integration Tests

End-to-end workflow tests for astrophysical applications of GWSNR. These tests validate complete scientific workflows and real-world use cases for gravitational wave detection and population studies.

## Test Files Overview

### `test_bbh_horizon_distance.py`
**BBH horizon distance calculations for detector sensitivity estimates**

**Scientific Context:**
Horizon distance represents the maximum luminosity distance at which a gravitational wave signal can be detected above threshold. This is crucial for:
- Detector sensitivity characterization
- Observing run planning
- Expected detection rate predictions

**Test Coverage:**
- **Numerical Method**: Sky location optimization with antenna response maximization
- **Analytical Method**: Rapid calculation using interpolation techniques  
- **Equal-mass BBH System**: 30+30 M☉ representative system
- **Output Validation**: Astrophysical range (1-10 kMpc), data types, numerical properties
- **Performance**: Sub-minute calculation times
- **Reproducibility**: Consistent results across multiple runs
- **Error Handling**: Invalid input type detection

**Key Validations:**
- Horizon distances within expected astrophysical ranges
- Sky location optimization produces maximum antenna response
- Numerical and analytical methods show consistency
- Performance suitable for operational use

### `test_bbh_population_detectable_fraction.py`
**Population-level detectable fraction calculations for selection effects**

**Scientific Context:**
The detectable fraction P(λ|SNR_th) represents the selection effect function - the fraction of a population detectable above threshold. Essential for:
- Population inference corrections
- Rate calculation normalization  
- Bias quantification in detected samples

**Test Coverage:**
- **Selection Effect Function**: P(λ|SNR_th) calculation using Pdet methods
- **Astrophysical Population**: 10,000 BBH events from LER package
- **Detection Thresholds**: Network SNR ≥ 10 detection criteria
- **Statistical Validation**: Boolean detection probability calculations
- **Performance**: Large population handling (10k+ events)
- **Output Validation**: Detectable fractions in realistic ranges (0-1)

**Key Validations:**
- Detectable fractions within physically reasonable bounds
- Consistent results for population-scale calculations
- Performance suitable for large astrophysical populations
- Integration with realistic BBH parameter distributions

### `test_bbh_rate.py`
**End-to-end BBH merger rate calculations**

**Scientific Context:**
BBH merger rate estimation combines intrinsic astrophysical rates with gravitational wave detectability:
- R = R_intrinsic × (detectable_events / total_events)
- Critical for cosmological and astrophysical inference
- Validates complete detection pipeline functionality

**Test Coverage:**
- **Hybrid SNR Method**: Interpolation + selective inner product recalculation
- **Rate Calculation Pipeline**: Complete workflow from parameters to rates
- **Astrophysical Parameters**: Realistic BBH mass, spin, distance distributions
- **Detection Efficiency**: Network SNR ≥ 8 threshold application
- **Performance Optimization**: Efficient handling of large event samples
- **Validation**: Rate estimates within expected astrophysical ranges

**Key Validations:**
- End-to-end pipeline produces realistic merger rates
- Hybrid approach balances accuracy and computational efficiency
- Performance suitable for population-scale studies
- Results consistent with observational constraints

## Shared Resources

### `bbh_gw_params.json`
**Astrophysical BBH parameter dataset**
- **Source**: Generated using LER (Lens Einstein Ring) package
- **Size**: 10,000 realistic BBH events
- **Parameters**: Masses, spins, distances, sky locations, orientation angles
- **Usage**: Provides realistic parameter distributions for integration testing
- **Astrophysical Realism**: Follows observed BBH population characteristics

## Running Integration Tests

### All Integration Tests
```bash
pytest tests/integration/ -v -s
```

### Individual Test Files
```bash
# Horizon distance calculations
pytest tests/integration/test_bbh_horizon_distance.py -v -s

# Population detectable fractions  
pytest tests/integration/test_bbh_population_detectable_fraction.py -v -s

# Merger rate calculations
pytest tests/integration/test_bbh_rate.py -v -s
```

### Specific Test Methods
```bash
# Numerical horizon distance
pytest tests/integration/test_bbh_horizon_distance.py::TestBBHHorizonDistanceCalculation::test_horizon_distance_bbh_numerical -v -s

# Analytical horizon distance
pytest tests/integration/test_bbh_horizon_distance.py::TestBBHHorizonDistanceCalculation::test_horizon_distance_bbh_analytical -v -s

# Population detectable fraction
pytest tests/integration/test_bbh_population_detectable_fraction.py::TestBBHSelectionEffect::test_detectable_fraction_bbh -v -s

# Rate calculation
pytest tests/integration/test_bbh_rate.py::TestBBHRateCalculation::test_rate_bbh -v -s
```

## Scientific Applications

### **Detector Characterization**
- Sensitivity range estimates for current and future detectors
- Observing run planning and optimization
- Expected detection rate predictions

### **Population Studies**  
- Selection effect quantification for population inference
- Bias correction in detected BBH samples
- Completeness estimates for observational catalogs

### **Rate Calculations**
- BBH merger rate estimates from gravitational wave observations
- Cosmological parameter inference support
- Astrophysical population model validation

## Validation Criteria

### **Astrophysical Realism**
- Results within observed/theoretical ranges
- Consistency with LIGO-Virgo observations
- Physically meaningful parameter relationships

### **Computational Performance**
- Suitable for operational/research workflows
- Scalable to large populations (10k+ events)
- Reasonable execution times (minutes, not hours)

### **Scientific Accuracy**
- Cross-validation with established methods
- Proper statistical treatment of uncertainties
- Appropriate approximations for intended use cases

## Expected Performance

<!-- ### **Execution Times**
- **Horizon Distance**: 30-60 seconds per calculation
- **Detectable Fraction**: 1-5 minutes for 10k events  
- **Rate Calculation**: 2-10 minutes for complete workflow -->
- **Full Integration Suite**: within 5 minutes total

<!-- ### **Resource Requirements**
- **Memory**: 1-4 GB for large populations
- **CPU**: Efficient multiprocessing utilization
- **Storage**: Minimal (test data ~10-50 MB) -->

## Dependencies

### **Core Requirements**
- `gwsnr`, `pytest`
- Test data files (`bbh_gw_params.json`)

### **Scientific Context**
- Understanding of gravitational wave detection principles
- Familiarity with BBH population characteristics
- Knowledge of selection effects in astronomical surveys

## Integration with GWSNR Ecosystem

These tests validate GWSNR's role in:
- **LIGO-Virgo-KAGRA data analysis pipelines**
- **Population inference frameworks**
- **Detector network optimization studies**
- **Next-generation detector planning**
- **Multi-messenger astrophysics applications**

The integration tests ensure GWSNR produces scientifically meaningful results for real-world gravitational wave astronomy applications.