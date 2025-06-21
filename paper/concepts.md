

### Mathematical Formulation

The `gwsnr` package provides two efficient methods for computing the optimal SNR in gravitational-wave data analysis: Noise-Weighted Inner Product Method with Multiprocessing and Partial Scaling Interpolation Method, and two others for estimating Pdet for precessing systems: ANN-based Pdet Estimation and Partial Scaling Interpolation Method with SNR recalculation. Extensive details of these methods can be found in the package documentation (@gwsnr:documentation).

#### Noise-Weighted Inner Product Method with Multiprocessing

Noise-weighted inner product, a traditionally robust technique suitable for any frequency-domain gravitational waveform, including complex models with spin precession and higher-order harmonics available in `lalsimulation` (@lalsuite). Following @Allen:2012, the inner product between two frequency-domain signals, $\tilde{a}(f)$ and $\tilde{b}(f)$, is defined as:

$$\left< a | b \right> = 4 \Re \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df$$

Here, $S_n(f)$ is the one-sided power spectral density of the detector noise and $[f_{\min}, f_{\max}]$ is the analysis frequency band. 
The optimal SNR, $\rho$, is the norm of the signal, $\rho = \sqrt{\langle h | h \rangle}$. 
For a gravitational wave signal composed of plus ($h_+$) and cross ($h_\times$) polarizations, and assuming orthogonality between them, the SNR can be expressed in terms of the detector's antenna patterns, $F_+$ and $F_\times$:

$$\rho = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_{\times}^2 \left< \tilde{h}_{\times} | \tilde{h}_{\times} \right> }$$

This calculation, while versatile, can be computationally demanding, primarily due to waveform generation. `gwsnr` mitigates this by parallelizing waveform generation across multiple CPU cores and accelerating the antenna pattern and inner product calculations using `numba.njit` compilation. Additionally, the package offers optional support for JAX-based acceleration through the `ripple` waveform library [@Edwards:2023sak], using `jax.jit` for compilation and `jax.vmap` for batched operations. 

#### Partial Scaling Interpolation Method

For non-spinning or aligned-spin binary systems, `gwsnr` implements a highly efficient interpolation-based technique called the Partial Scaling method. This approach, adapted from the FINDCHIRP algorithm (@Allen:2012), decouples the computationally expensive parts of the SNR calculation from the extrinsic source parameters. It achieves this by defining a "partial scaled SNR," $\rho_{1/2}$, which isolates the dependency on the intrinsic parameters (masses and spins). For a given full IMR waveform SNR, $\rho_{\text{full}}$, the partial SNR is defined as:

$$\rho_{1/2} = \left(\frac{D_\mathrm{eff}}{1~\mathrm{Mpc}}\right) \left(\frac{\mathcal{M}}{M_\odot}\right)^{-5/6} \times \rho_{\text{full}}$$

Here, $\mathcal{M}$ is the chirp mass and $D_{\text{eff}}$ is the effective distance, which encapsulates the luminosity distance, sky location, and binary orientation. Since $\rho_{1/2}$ depends only on the intrinsic properties of the binary, its value can be pre-computed on a grid and stored. For non-spinning systems, this is a two-dimensional grid of total mass ($M$) and mass ratio ($q$), while for aligned-spin systems, it is a four-dimensional grid that also includes the two spin magnitudes. To find the SNR for a new binary, `gwsnr` performs a rapid cubic spline interpolation on the pre-computed grid to find the corresponding $\rho_{1/2}$ value. The final SNR is then recovered almost instantaneously by applying the scaling transformation:

$$\rho = \rho_{1/2} \times \left(\frac{\mathcal{M}}{M_\odot}\right)^{5/6} \times \left(\frac{1~\mathrm{Mpc}}{D_\mathrm{eff}}\right)$$

This procedure transforms a computationally intensive integration into a simple, JIT-compiled table lookup and multiplication, enabling massive performance gains for large-scale population studies.


#### ANN-based Pdet Estimation

The *`gwsnr`* package now incorporates an artificial neural network (ANN) model, developed using *`TensorFlow`* (@tensorflow2015) and *`sklearn`* (@scikitlearn), to rapidly estimate the Pdet in binary black hole (BBH) systems using the IMRPhenomXPHM waveform approximant. This complex IMR waveform model accounts for spin-precessing systems with subdominant harmonics. The ANN model is especially useful when precise signal-to-noise ratio (SNR) calculations are not critical, providing a quick and effective means of estimating Pdet. This value indicates detectability under Gaussian noise by determining if the SNR exceeds a certain threshold (e.g. $\rho_{\rm th}=8$). Trained on a large dataset from the *`ler`* package, the ANN model uses 'partial scaled SNR' values as a primary input, reducing input dimensionality from 15 to 5 and enhancing accuracy. This approach offers a practical solution for assessing detectability under specified conditions. Other similar efforts with ANN models are detailed in @ChapmanBird2023, @Gerosa:2020, @Callister2024 etc.

In addition to providing trained ANN models for specific configurations, *`gwsnr`* offers users the flexibility to develop and train custom models tailored to their unique requirements. This adaptability allows for optimization based on variations in detector sensitivity, gravitational wave properties, and other research-specific factors, ensuring maximum model effectiveness across different scenarios.

#### Partial Scaling Interpolation Method with SNR Recalculation for Pdet Estimation

While the Partial Scaling method is highly efficient aligned-spin systems, we can further enhance its utility by recalculating the SNR for precessing systems in predefined samll range of the genrerated SNRs. This is done by first obtaining optimal SNRs with the Partial Scaling method, picking the SNRs nearby to $\rho_{\rm th}$ and then recalculating the SNRs for these systems using the Noise-Weighted Inner Product Method. This approach allows us to leverage the speed of the Partial Scaling method while ensuring that we have accurate SNR values for systems that are close to the detection threshold. The recalculated SNRs can then be used to estimate Pdet, providing a balance between computational efficiency and accuracy.


Below is a **minimally revised version** of your text, with corrected grammar and improved sentence structure for clarity. I’ve added a few comments about possible incoherence or ambiguity where relevant.

---

### Mathematical Formulation

The `gwsnr` package provides two efficient methods for computing the optimal SNR in gravitational-wave data analysis: the Noise-Weighted Inner Product Method with Multiprocessing and the Partial Scaling Interpolation Method. In addition, there are two approaches for estimating $P_{\rm det}$ for precessing systems: ANN-based $P_{\rm det}$ Estimation and the Partial Scaling Interpolation Method with SNR recalculation. Extensive details of these methods can be found in the package documentation [@gwsnr\:documentation].

#### Noise-Weighted Inner Product Method with Multiprocessing

The noise-weighted inner product is a robust and widely used technique, suitable for any frequency-domain gravitational waveform, including complex models with spin precession and higher-order harmonics available in `lalsimulation` [@lalsuite]. Following [@Allen:2012], the inner product between two frequency-domain signals, $\tilde{a}(f)$ and $\tilde{b}(f)$, is defined as:

$$
\left< a | b \right> = 4 \Re \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df
$$

Here, $S_n(f)$ is the one-sided power spectral density of the detector noise, and $[f_{\min}, f_{\max}]$ is the analysis frequency band.
The optimal SNR, $\rho$, is the norm of the signal: $\rho = \sqrt{\langle h | h \rangle}$.
For a gravitational wave signal composed of plus ($h_+$) and cross ($h_\times$) polarizations, and assuming orthogonality between them, the SNR can be expressed in terms of the detector's antenna patterns, $F_+$ and $F_\times$:

$$
\rho = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_{\times}^2 \left< \tilde{h}_{\times} | \tilde{h}_{\times} \right> }
$$

This calculation, while versatile, can be computationally demanding, primarily due to waveform generation. `gwsnr` mitigates this by parallelizing waveform generation across multiple CPU cores and accelerating the antenna pattern and inner product calculations using `numba.njit` compilation. Additionally, the package offers optional support for JAX-based acceleration through the `ripple` waveform library [@Edwards:2023sak], using `jax.jit` for compilation and `jax.vmap` for batched operations.

#### Partial Scaling Interpolation Method

For non-spinning or aligned-spin binary systems, `gwsnr` implements a highly efficient interpolation-based technique called the Partial Scaling method. This approach, adapted from the FINDCHIRP algorithm [@Allen:2012], decouples the computationally expensive parts of the SNR calculation from the extrinsic source parameters. It achieves this by defining a "partial scaled SNR," $\rho_{1/2}$, which isolates the dependence on the intrinsic parameters (masses and spins). For a given full IMR waveform SNR, $\rho_{\text{full}}$, the partial SNR is defined as:

$$
\rho_{1/2} = \left(\frac{D_\mathrm{eff}}{1~\mathrm{Mpc}}\right) \left(\frac{\mathcal{M}}{M_\odot}\right)^{-5/6} \times \rho_{\text{full}}
$$

Here, $\mathcal{M}$ is the chirp mass and $D_{\text{eff}}$ is the effective distance, which encapsulates the luminosity distance, sky location, and binary orientation. Since $\rho_{1/2}$ depends only on the intrinsic properties of the binary, its value can be pre-computed on a grid and stored. For non-spinning systems, this is a two-dimensional grid of total mass ($M$) and mass ratio ($q$), while for aligned-spin systems, it is a four-dimensional grid that also includes the two spin magnitudes. To find the SNR for a new binary, `gwsnr` performs a rapid cubic spline interpolation on the pre-computed grid to find the corresponding $\rho_{1/2}$ value. The final SNR is then recovered almost instantaneously by applying the scaling transformation:

$$
\rho = \rho_{1/2} \times \left(\frac{\mathcal{M}}{M_\odot}\right)^{5/6} \times \left(\frac{1~\mathrm{Mpc}}{D_\mathrm{eff}}\right)
$$

This procedure transforms a computationally intensive integration into a simple, JIT-compiled table lookup and multiplication, enabling massive performance gains for large-scale population studies.

#### ANN-based Pdet Estimation

The `gwsnr` package now incorporates an artificial neural network (ANN) model, developed using TensorFlow [@tensorflow2015] and scikit-learn [@scikitlearn], to rapidly estimate $P_{\rm det}$ in binary black hole (BBH) systems using the IMRPhenomXPHM waveform approximant. This complex IMR waveform model accounts for spin-precessing systems with subdominant harmonics. The ANN model is especially useful when precise signal-to-noise ratio (SNR) calculations are not critical, providing a quick and effective means of estimating $P_{\rm det}$. This value indicates detectability under Gaussian noise by determining if the SNR exceeds a certain threshold (e.g., $\rho_{\rm th}=8$). Trained on a large dataset from the `ler` package, the ANN model uses 'partial scaled SNR' values as a primary input, reducing input dimensionality from 15 to 5 and enhancing accuracy. This approach offers a practical solution for assessing detectability under specified conditions. Other similar efforts with ANN models are detailed in [@ChapmanBird2023, @Gerosa:2020, @Callister2024].

In addition to providing trained ANN models for specific configurations, `gwsnr` offers users the flexibility to develop and train custom models tailored to their unique requirements. This adaptability allows for optimization based on variations in detector sensitivity, gravitational-wave properties, and other research-specific factors, ensuring maximum model effectiveness across different scenarios.

#### Partial Scaling Interpolation Method with SNR Recalculation for Pdet Estimation

While the Partial Scaling method is highly efficient for aligned-spin systems, its utility can be further enhanced by recalculating the SNR for precessing systems within a predefined small range of generated SNRs. This is done by first obtaining optimal SNRs with the Partial Scaling method, selecting the SNRs near $\rho_{\rm th}$, and then recalculating the SNRs for these systems using the Noise-Weighted Inner Product Method. This approach allows us to leverage the speed of the Partial Scaling method while ensuring accurate SNR values for systems close to the detection threshold. The recalculated SNRs can then be used to estimate $P_{\rm det}$, providing a balance between computational efficiency and accuracy.
