
# Noise-Weighted Inner Product Method

The `gwsnr` package implements the standard noise-weighted inner product method to compute the signal-to-noise ratio (SNR), a core technique in gravitational-wave data analysis. This method is particularly suitable for frequency-domain waveforms generated by the `lalsimulation` library [@lalsuite], including complex models that incorporate spin precession and subdominant harmonic modes.

To enhance computational performance, `gwsnr` parallelizes waveform generation across multiple CPU cores and uses `numba`'s Just-In-Time (`njit`) compilation to accelerate antenna response functions, chirptime (signal duration) calculation, and inner product calculations. Additionally, the package offers optional support for JAX-based acceleration through the `ripple` waveform library [@Edwards:2023sak], using `jax.jit` for compilation and `jax.vmap` for batched operations.

## Mathematical Formulation

Following [@Allen:2012](https://arxiv.org/pdf/gr-qc/0509116), the noise-weighted inner product between two complex-valued frequency-domain waveforms, $\tilde{a}(f)$ and $\tilde{b}(f)$, defined as:

$$
\begin{align}
\left< a | b \right> = 4 \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}(f), \tilde{b}^*(f)}{S_n(f)} , df \tag{1}
\end{align}
$$

Here, $S_n(f)$ denotes the one-sided power spectral density (PSD) of the detector noise, and $[f_{\min}, f_{\max}]$ is the analysis frequency band.

GW waveform can be expressed as a combination of plus and cross polarizations, $\tilde{h}(f) = \tilde{h}_+(f) + i\, \tilde{h}_\times(f)$. Using this, we can write optimal SNR as:

$$
\begin{align}
\rho = \sqrt{ \left< h | h \right> } = \sqrt{ 4 \int_{f_{\min}}^{f_{\max}} \frac{\tilde{h}(f)\, \tilde{h}^*(f)}{S_n(f)} \, df } \tag{2}
\end{align}
$$

Assuming that $\tilde{h}+$ and $\tilde{h}\times$ are orthogonal, which is a good approximation for most waveform models including precessing cases, the SNR can be written in terms of detector response functions:

$$
\begin{align}
\rho = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_\times^2 \left< \tilde{h}_\times | \tilde{h}_\times \right> } \tag{3}
\end{align}
$$

The antenna pattern functions $F_+$ and $F_\times$ depend on the detector's orientation and the source's sky location ($\alpha$, $\delta$), inclination ($\iota$), polarization angle ($\psi$), and geocentric time ($t_c$). The waveform polarizations $\tilde{h}_{+,\times}$ themselves depend on both intrinsic parameters (component masses $m_1$, $m_2$, and spins) and extrinsic parameters (luminosity distance $D_L$, inclination angle $\iota$, coalescence phase $\phi_c$, and $t_c$).

This method provides a flexible and robust framework for computing optimal SNRs using arbitrary frequency-domain waveform models and detector configurations. The bottleneck in this method is waveform generation, which is parallelized across multiple CPU cores. The remaining computations are efficiently handled through `njit` compilation and acceleration, making this approach more efficient than traditional Bilby SNR generation.