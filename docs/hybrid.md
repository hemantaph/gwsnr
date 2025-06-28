# Hybrid Strategy for Spin-Precessing Systems

The `gwsnr` package extends its interpolation capabilities to handle spin-precessing systems through a hybrid approach that combines computational efficiency with enhanced accuracy. For systems with non-aligned spins (precession and higher-order modes), the package employs the IMRPhenomXPHM waveform model while leveraging aligned-spin approximations for initial interpolation.

### Spin Projection Method

For interpolation purposes, `gwsnr` projects the spin vectors onto the orbital angular momentum axis, computing effective aligned-spin components:

$$a_{1,z} = a_1 \cos(\theta_{1,z}) \quad \text{and} \quad a_{2,z} = a_2 \cos(\theta_{2,z})$$

where $\theta_{1,z}$ and $\theta_{2,z}$ are the tilt angles between the individual spin vectors and the orbital angular momentum vector.

### Hybrid Recalculation Strategy

The SNR recalculation method implements a two-stage computational approach:

1. **Initial Estimation**: Rapidly compute SNRs using the interpolation method with projected aligned spins
2. **Selective Recalculation**: For systems near the detection threshold (typically SNR ∈ [6,8]), recalculate SNRs using the full inner product method with the IMRPhenomXPHM waveform model

This hybrid strategy optimizes computational resources by applying the more expensive full waveform calculation only to borderline detectable events, where accurate SNR determination is most critical for detection decisions. The majority of events—those clearly above or below the detection threshold—benefit from the speed of interpolation without compromising overall accuracy.

### Implementation Benefits

- **Computational Efficiency**: Reduces overall computation time by orders of magnitude compared to full inner product calculations
- **Enhanced Accuracy**: Maintains high precision for detection-critical events through selective recalculation
- **Flexible Thresholds**: Allows user-defined SNR ranges for triggering recalculation based on specific analysis requirements
- **Precession Handling**: Accommodates complex spin dynamics while preserving computational tractability