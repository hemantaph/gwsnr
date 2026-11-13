# Noise-Weighted Inner Product Method

The inner-product method computes the optimal signal-to-noise ratio (SNR) of a specified gravitational-wave signal using a detector's one-sided noise power spectral density (PSD). It is a deterministic signal norm, not a matched-filter search over noisy data. Each detector's noise is assumed stationary, and network SNRs assume uncorrelated detector noise.

`snr_method="inner_product"` generates frequency-domain polarizations through Bilby and LALSimulation. It supports precessing and nonprecessing waveform models, subject to the chosen approximant's parameter restrictions. Waveform generation is distributed across CPU processes; Numba accelerates the antenna-response and inner-product calculations. The optional `inner_product_jax` backend generates waveforms with Ripple and uses the same full polarization expansion.

## Complex and real inner-product conventions

The low-level routine `noise_weighted_inner_product(a, b, psd, duration)` returns the **complex** noise-weighted product

$$
I(a,b)=4\int_{f_{\min}}^{f_{\max}}
\frac{\tilde a^*(f)\tilde b(f)}{S_n(f)}\,df
\simeq \frac{4}{T}\sum_k\frac{\tilde a_k^*\tilde b_k}{S_{n,k}},
$$

where $T$ is the waveform duration and $\Delta f=1/T$. The sum runs over the analysis band with valid PSD values. With a real PSD, $I(b,a)=I(a,b)^*$. The corresponding real noise-weighted inner product is $(a|b)=\operatorname{Re}I(a,b)$.

For a template $h$ and detector data $d$, one complex matched-filter convention is $I(h,d)/\sqrt{I(h,h)}$. Its real part gives the statistic for a fixed template phase; taking its magnitude corresponds to maximizing over the overall phase of that template. GWSNR's direct optimal-SNR calculation uses the signal norm $\rho_{\rm opt}=\sqrt{I(h,h)}$ and does not require a noise realization. The `observed_snr` mode of the direct inner-product method is not implemented.

## Full detector response and optimal SNR

For the real, frequency-independent antenna factors used here, the detector strain is

$$
\tilde h(f)=F_+\tilde h_+(f)+F_\times\tilde h_\times(f).
$$

The antenna factors depend on the detector geometry, sky position, polarization angle, and geocentric time. Inclination enters the waveform polarizations, not the antenna factors. A common frequency-domain arrival-time phase multiplies both terms and cancels in the norm.

Expanding the detector norm gives **all four terms**:

$$
\begin{aligned}
\rho_{\rm opt}^2
&=F_+^2 I(h_+,h_+)+F_\times^2 I(h_\times,h_\times)\\
&\quad+F_+F_\times I(h_+,h_\times)
+F_\times F_+ I(h_\times,h_+).
\end{aligned}
$$

The two mixed products are complex conjugates, so this is equivalently

$$
\boxed{\rho_{\rm opt}^2=
F_+^2 I(h_+,h_+)+F_\times^2 I(h_\times,h_\times)
+2F_+F_\times\operatorname{Re}I(h_+,h_\times).}
$$

The mixed contribution can be positive or negative; its sign must be retained. The complete norm is real and nonnegative.

### Nonprecessing and nonorthogonal polarizations

The two-term expression

$$
\rho_{\rm opt}^2=F_+^2 I(h_+,h_+)+F_\times^2 I(h_\times,h_\times)
$$

is valid when $\operatorname{Re}I(h_+,h_\times)=0$ for the selected PSD and frequency band (or when $F_+F_\times=0$ for that detector). This is orthogonality under the **real** inner product; the complex product $I(h_+,h_\times)$ need not be zero.

For the nonprecessing, dominant-quadrupole `IMRPhenomD` waveform, the two polarizations are in quadrature and the real mixed product vanishes up to numerical precision. Nonprecession alone is not a general justification for dropping the mixed terms: higher-mode content can change this relation. For precessing or higher-mode waveforms such as `IMRPhenomXPHM`, compute the mixed product explicitly. It can be nonzero, but need not be nonzero for every source orientation or detector.

### Implementation and release scope

Since **version 0.5.2**, all direct inner-product workers return `hp_inner_hp`, `hc_inner_hc`, and the complex `hp_inner_hc`. The reverse mixed product is its complex conjugate. Both the LAL and Ripple SNR paths combine them as

```python
rho_squared = (
    Fp**2 * hp_inner_hp
    + Fc**2 * hc_inner_hc
    + 2.0 * Fp * Fc * np.real(hp_inner_hc)
)
```

The worker arrays have shape `(number_of_detectors, number_of_events)` after collection. Their PSD weighting, frequency grid, and duration are identical for all three products. Legacy workers and waveform-failure returns include the additional mixed product as well; code directly unpacking these helper return tuples must allow for it. The public SNR dictionary keys are unchanged.

Hybrid SNR recalculation calls the corrected LAL method. Detection probabilities and numerical horizon calculations also use the corrected SNR when configured with the direct inner-product backend. Fast interpolation retains its nonprecessing quadrupole scaling. Existing ANN weights, cached SNRs, and externally supplied training labels are not retroactively corrected; regenerate affected labels and retrain ANN models if they used the former two-term result. See [ANN](ann.md) and [Hybrid](hybrid.md).

The current Ripple interface supports `IMRPhenomD`, `IMRPhenomXAS`, `TaylorF2`, and `IMRPhenomD_NRTidalv2`. It does not expose `IMRPhenomXPHM`; use the LAL backend for that model. To run JAX on CPU, set `JAX_PLATFORMS=cpu` before importing JAX. For double-precision comparisons, also set `JAX_ENABLE_X64=True`.

## Network optimal SNR

For independent detector noise,

$$
\rho_{\rm opt,net}=\sqrt{\sum_i\rho_{{\rm opt},i}^2}.
$$

Each detector's norm includes its own antenna factors and PSD-weighted mixed product.

## Consistency checks against Bilby

A meaningful comparison requires identical source parameters, waveform approximant and source model, reference and minimum frequencies, sampling frequency, duration, detector geometry, and PSDs. Supply ASD files as ASD, not PSD. Use `fixed_duration` to match Bilby's duration explicitly; otherwise GWSNR estimates the duration from the source masses. Set Bilby's maximum frequency to the Nyquist frequency used by GWSNR.

Compare against `ifo.optimal_snr_squared(ifo.get_detector_response(polarizations, parameters))`, with zero-noise detector data, unit window factor, and identity calibration. For `IMRPhenomD`, provide supported aligned-spin parameters: LAL rejects nonzero transverse spins.

The repository's `examples/snr_precessing_spin.ipynb` tests `IMRPhenomD` and `IMRPhenomXPHM`, audits the settings, and reconstructs the full detector norm. For the supplied precessing event, the corrected network SNR is approximately **50.00452642**, compared with Bilby's **50.00452694**. The former two-term result was **50.18093445**. The small remaining difference is explained by GWSNR's GPS-to-GMST approximation in its antenna factors. With identical antenna factors, the comparison passes at $10^{-12}$ relative tolerance. These results describe this test event, not a population-wide error bound.

## Precessing-waveform example

Run this as a script, keeping the multiprocessing entry-point guard:

```python
import numpy as np
from gwsnr import GWSNR

if __name__ == "__main__":
    calculator = GWSNR(
        snr_method="inner_product",
        waveform_approximant="IMRPhenomXPHM",
        npool=1,
        sampling_frequency=1024.0,
        minimum_frequency=20.0,
        reference_frequency=20.0,
        fixed_duration=64.0,
        duration_max=64.0,
        psds={
            "L1": "aLIGO_O4_high_asd.txt",
            "H1": "aLIGO_O4_high_asd.txt",
            "V1": "AdV_asd.txt",
        },
    )
    parameters = dict(
        mass_1=np.array([36.0]),          # detector-frame solar masses
        mass_2=np.array([29.0]),
        luminosity_distance=np.array([440.0]),  # Mpc
        theta_jn=np.array([1.0]),        # angle between J and line of sight, radians
        ra=np.array([3.435]),
        dec=np.array([-0.408]),
        psi=np.array([0.0]),
        phase=np.array([0.0]),
        geocent_time=np.array([1126259462.4]),   # GPS seconds
        a_1=np.array([0.3]),
        a_2=np.array([0.2]),
        tilt_1=np.array([0.5]),
        tilt_2=np.array([0.8]),
        phi_12=np.array([0.0]),
        phi_jl=np.array([0.0]),
    )
    snrs = calculator.optimal_snr(gw_param_dict=parameters)
    print(snrs)  # optimal_snr_L1, optimal_snr_H1, optimal_snr_V1, optimal_snr_net
```

Waveform generation is generally the main cost. Runtime depends on the waveform model, duration, number of sources, and CPU-process count. The mixed term adds one complex polarization inner product per detector and source; it does not require a second waveform generation.
