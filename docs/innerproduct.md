
# Noise-Weighted Inner Product Method

The `gwsnr` package implements the standard noise-weighted inner product method to compute the signal-to-noise ratio (SNR), a core technique in gravitational-wave data analysis. This method is particularly suitable for frequency-domain waveforms generated by the [lalsimulation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation.html) library, including complex models that incorporate spin precession and subdominant harmonic modes (e.g. IMRPhenomXHM).

To enhance computational performance, `gwsnr` parallelizes waveform generation across multiple CPU cores and uses `numba`'s Just-In-Time (`njit`) compilation to accelerate antenna response functions, chirptime (signal duration) calculation, and inner product calculations. Additionally, the package offers optional support for JAX-based acceleration through the [ripple](https://github.com/hemantaph/ripple) waveform library, using `jax.jit` for compilation and `jax.vmap` for batched operations.

## Mathematical Formulation

Following [@Allen:2012](https://arxiv.org/pdf/gr-qc/0509116), the noise-weighted inner product between two complex-valued frequency-domain waveforms, $\tilde{a}(f)$ and $\tilde{b}(f)$, defined as

$$
\begin{align}
\left< a | b \right> = 4 \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}(f), \tilde{b}^*(f)}{S_n(f)} , df \tag{1}
\end{align}
$$

Here, $S_n(f)$ denotes the one-sided power spectral density (PSD) of the detector noise, and $[f_{\min}, f_{\max}]$ is the analysis frequency band.

### Optimal SNR 

GW waveform can be expressed as a combination of plus and cross polarizations, $\tilde{h}(f) = \tilde{h}_+(f) + i\, \tilde{h}_\times(f)$. Using this, we can write optimal SNR as:

$$
\begin{align}
\rho_{\rm opt} = \sqrt{ \left< h | h \right> } = \sqrt{ 4 \int_{f_{\min}}^{f_{\max}} \frac{\tilde{h}(f)\, \tilde{h}^*(f)}{S_n(f)} \, df } \tag{2}
\end{align}
$$

Assuming that $\tilde{h}_+$ and $\tilde{h}_\times$ are orthogonal, which is a good approximation for most waveform models including precessing cases, the SNR can be written in terms of detector response functions, and it reads

$$
\begin{align}
\rho_{\rm opt} = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_\times^2 \left< \tilde{h}_\times | \tilde{h}_\times \right> } \tag{3}
\end{align}
$$

The antenna pattern functions $F_+$ and $F_\times$ depend on the detector's orientation and the source's sky location ($\alpha$, $\delta$), inclination ($\iota$), polarization angle ($\psi$), and geocent time ($t_c$). The waveform polarizations $\tilde{h}_{+,\times}$ themselves depend on both intrinsic parameters (component masses $m_1$, $m_2$, and spins) and extrinsic parameters (luminosity distance $d_L$, inclination angle $\iota$, coalescence phase $\phi_c$, and $t_c$).


### Network Optimal SNR

The network optimal SNR, $\rho_{\rm opt, net}$, combines the individual detector optimal SNRs to provide a single measure of detectability across the entire detector network. For a network of $n$ detectors, the network SNR is calculated as:

$$
\begin{align}
\rho_{\rm opt, net} = \sqrt{\sum_{i=1}^{n} \rho_{{\rm opt,} i}^2} \tag{4}
\end{align}
$$

where $\rho_{{\rm opt,} i}$ is the optimal SNR in the $i$-th detector. This quadrature sum assumes that the noise in different detectors is uncorrelated, which is a reasonable approximation for geographically separated gravitational-wave detectors.

### Matched Filter SNR

---

Noise-Weighted Inner Product Method provides a flexible and robust framework for computing optimal SNRs using arbitrary frequency-domain waveform models and detector configurations. The bottleneck in this method is waveform generation, which is mitigated to some extent by parallelizing it across multiple CPU cores. The remaining computations are efficiently handled through `numba.njit` (with multithreaded `prange` loops) compilation and acceleration, making this approach more efficient than traditional Bilby SNR generation.

## Example Usage

```python
# loading GWSNR class from the gwsnr package
import gwsnr
import numpy as np

# initializing the GWSNR class with inner product as the signal-to-noise ratio type
# IMRPhenomXPHM precessing waveform approximant is used
gwsnr = gwsnr.GWSNR(snr_type='inner_product', waveform_approximant='IMRPhenomXPHM')

# defining the parameters for the gravitational wave signal for a BBH with GW150914 like parameters
param_dict= dict(
    mass_1=np.array([36.0]), # mass of the primary black hole in solar masses
    mass_2=np.array([29.0]), # mass of the secondary black hole in solar masses
    luminosity_distance=np.array([440.0]), # luminosity distance in Mpc
    theta_jn=np.array([1.0]), # inclination angle in radians
    ra=np.array([3.435]), # right ascension in radians
    dec=np.array([-0.408]), # declination in radians
    psi=np.array([0.0]),  # polarization angle in radians
    geocent_time=np.array([1126259462.4]), # geocentric time in GPS seconds
    a_1=np.array([0.3]), # dimensionless spin of the primary black hole
    a_2=np.array([0.2]), # dimensionless spin of the secondary black hole
    tilt_1=np.array([0.5]), # tilt angle of the primary black hole in radians
    tilt_2=np.array([0.8]), # tilt angle of the secondary black hole in radians
    phi_12=np.array([0.0]), # Relative angle between the primary and secondary spin of the binary in radians
    phi_jl=np.array([0.0]), # Angle between the total angular momentum and the orbital angular momentum in radians
)

# signal-to-noise ratio with detectors LIGO-Hanford, LIGO-Livingston, and Virgo with O4 observing run sensitivity
snrs = gwsnr.snr(**param_dict)

print('Computed SNRs with inner product:\n', snrs)
```

**Expected Output:**

```
Computed SNRs with inner product:
{'L1': array([46.53]), 'H1': array([48.20]), 'V1': array([13.21]), 'optimal_snr_net': array([68.28])}
```

## Performance

When testing with 10,000 BBH samples of IMRPhenomXPHM precessing-waveforms with O4 sensitivity, the noise-weighted inner product method achieves an average computation time of approximately 50 seconds on a single CPU core. This performance is significantly improved when using multiple cores, reducing the average time to around 8.5 seconds with 8 cores. Note that, while the waveform generation is parallelized across multiple CPU cores, the antenna pattern generation and inner product calculation itself is multithreaded using `numba.njit` with `prange` loops, which allows for efficient parallel computation across the available CPU cores and threads.
