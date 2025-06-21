---
title: '`gwsnr`: A python package for efficient signal-to-noise calculation of gravitational-waves'
tags:
  - Python
  - astrophysics
  - statistics
  - GWs
  - LIGO
authors:
  - name: Hemantakumar Phurailatpam
    orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Otto Akseli HANNUKSELA 
    affiliation: "1"
affiliations:
  - name: Department of Physics, The Chinese University of Hong Kong, Shatin, New Territories, Hong Kong
    index: 1
date: 12 January 2024
bibliography: paper.bib
---

# Summary 

Gravitational waves (GWs), ripples in spacetime predicted by Einstein's theory of General Relativity, have revolutionized astrophysics since their first detection in 2015 (@Abbott:2016, @Abbott:2016:pe). These waves are emitted by cataclysmic events like the merging of binary black holes (BBHs), binary neutron stars (BNSs) and BH-NS pairs, providing a unique window into the cosmos. A critical aspect of GW analysis is the Signal-to-Noise Ratio (SNR). SNR quantifies the strength of a GW signal relative to the background noise in a detector, like LIGO (@LIGO:2015, @Abbott:2020, @Buikema:2020), Virgo (@VIRGO:2015, @VIRGO:2019) or KAGRA (@Akutsu:2020, @Aso:2013). This ratio is pivotal in confirming the detection of GWs and extracting astrophysical information from them (@Abbott:2016:detection). However, specific scenarios in GW research, particularly in simulations of detectable GW events (@Abbott:2016:rates) and in hierarchical Bayesian analysis (@Thrane:2019) where selection effects are considered, demand extensive and efficient computation of SNR. This requirement presents a significant challenge, as conventional computational approaches, such as noise-weighted inner product, are typically time-consuming and impractical for such specialized and large-scale analyses (@Taylor:2018, @Gerosa:2020).

# Statement of Need

The *`gwsnr`* Python package addresses the need for efficient SNR computation in GW research. It provides a flexible and user-friendly interface, allowing users to combine various detector noise models, waveform models, detector configurations, and signal parameters. *`gwsnr`* enhances SNR calculations through several key features. Firstly, it utilizes an innovative interpolation method, employing a partial-scaling approach for accurately interpolating the SNR of GWs from spin-less and spin-aligned binary systems. Secondly, the package features a noise-weighted inner product method, similar to that in the *`bilby`* package (@Ashton:2019, @Ashton:2022), but enhanced with multiprocessing capabilities. This parallel processing is crucial for handling large datasets and computationally intensive analyses. Thirdly, a trained Artificial Neural Network (ANN) model is incorporated for rapid 'probability of detection' (Pdet) estimation for BBH systems with spin precession. Lastly, *`gwsnr`* leverages *`numpy`* (@numpy:2022) vectorization, and *`numba`*'s (@numba:2022) and *`JAX`*'s (@jax:2018) Just-In-Time compiler (`numbba.njit` and `jax.jit`), which optimizes performance by compiling Python code into machine code at runtime, drastically reducing execution times. This combination of advanced techniques and user-friendly design makes gwsnr a valuable tool for GW data analysis, particularly in simulating detectable compact binary mergers, determining rates of both lensed and unlensed GW events (as demonstrated by its use in the *`ler`* package; @ler:2024, @Leo:2024, @More:2025, @Janquart:2023, @Abbott:2021, @ligolensing:2023, @Wierda:2021, @Wempe:2022), and will help in the analysis of selection effects within hierarchical Bayesian frameworks (@Thrane:2019).

# Mathematical Formulation

The `gwsnr` package provides two efficient methods for computing the optimal SNR in GW data analysis: the Noise-Weighted Inner Product Method with Multiprocessing and the Partial Scaling Interpolation Method. In addition, there are two approaches for estimating $P_{\rm det}$ for precessing systems: ANN-based $P_{\rm det}$ Estimation and the Partial Scaling Interpolation Method with SNR recalculation. Extensive details of these methods can be found in the package documentation (@gwsnr:documentation).

### Noise-Weighted Inner Product Method with Multiprocessing

The noise-weighted inner product is a robust and widely used technique, suitable for any frequency-domain gravitational waveform, including complex models with spin precession and higher-order harmonics available in `lalsimulation` (@lalsuite:2018). Following (@Allen:2012), the inner product between two frequency-domain signals, $\tilde{a}(f)$ and $\tilde{b}(f)$, is defined as:

$$
\left< a | b \right> = 4 \Re \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df
$$

Here, $S_n(f)$ is the one-sided power spectral density of the detector noise, and $(f_{\min}, f_{\max})$ is the analysis frequency band.
The optimal SNR $\rho$, is the norm of the inner-product for the given signal $h$ : $\rho = \sqrt{\langle h | h \rangle}$.
For a gravitational wave signal composed of plus ($h_+$) and cross ($h_\times$) polarizations, and assuming orthogonality between them, the SNR can be expressed in terms of the detector's antenna patterns, $F_+$ and $F_\times$:

$$
\rho = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_{\times}^2 \left< \tilde{h}_{\times} | \tilde{h}_{\times} \right> }
$$

While this approach is versatile, it can be computationally intensive, with waveform generation representing the primary bottleneck. The `gwsnr` package addresses this challenge by parallelizing waveform generation across multiple CPU cores and accelerating the antenna pattern and inner product calculations using `numba.njit` compilation. Additionally, `gwsnr` provides optional support for JAX-based waveform generation and acceleration via the `ripple` waveform library [@Edwards:2023], utilizing `jax.jit` for just-in-time compilation and `jax.vmap` for efficient batched operations.

### Partial Scaling Interpolation Method

For non-spinning or aligned-spin binary systems restricted to the dominant harmonic mode, `gwsnr` implements a highly efficient interpolation-based technique called the Partial Scaling method. This approach, adapted from the FINDCHIRP algorithm (@Allen:2012), decouples the computationally expensive parts of the SNR calculation from the extrinsic source parameters. It achieves this by defining a "partial-scaled SNR" $\rho_{1/2}$, which isolates the dependence on the intrinsic parameters (masses and spins). For a given full IMR waveform SNR, $\rho_{\text{full}}$, the partial SNR is defined as:

$$
\rho_{1/2} = \left(\frac{D_\mathrm{eff}}{1~\mathrm{Mpc}}\right) \left(\frac{\mathcal{M}}{M_\odot}\right)^{-5/6} \times \rho_{\text{full}}
$$

Here, $\mathcal{M}$ is the chirp mass and $D_{\text{eff}}$ is the effective distance, which encapsulates the luminosity distance, sky location, and detector orientation wrt the binary. Since $\rho_{1/2}$ depends only on the intrinsic properties of the binary, its value can be pre-computed on a grid and stored. For non-spinning systems, this is a two-dimensional grid of total mass ($M$) and mass ratio ($q$), while for aligned-spin systems, it is a four-dimensional grid that also includes the two spin magnitudes. To find the SNR for a new binary, `gwsnr` performs a rapid cubic spline interpolation on the pre-computed grid to find the corresponding $\rho_{1/2}$ value. The final SNR is then recovered almost instantaneously by applying the scaling transformation:

$$
\rho = \rho_{1/2} \times \left(\frac{\mathcal{M}}{M_\odot}\right)^{5/6} \times \left(\frac{1~\mathrm{Mpc}}{D_\mathrm{eff}}\right)
$$

This procedure transforms a computationally intensive integration into a simple, JIT-compiled interpolation and multiplication, enabling massive performance gains for large-scale population studies.

### ANN-based Pdet Estimation

The `gwsnr` package now incorporates an artificial neural network (ANN) model, developed using TensorFlow (@tensorflow:2015) and scikit-learn (@scikitlearn:2011), to rapidly estimate $P_{\rm det}$ in binary black hole (BBH) systems using the IMRPhenomXPHM waveform approximant. This complex IMR waveform model accounts for spin-precessing systems with subdominant harmonics. The ANN model is especially useful when precise signal-to-noise ratio (SNR) calculations are not critical, providing a quick and effective means of estimating $P_{\rm det}$. This value indicates detectability under Gaussian noise by determining if the SNR exceeds a certain threshold (e.g., $\rho_{\rm th}=8$). Trained on a large dataset from the `ler` package, the ANN model uses 'partial scaled SNR' values as a primary input, reducing input dimensionality from 15 to 5 and enhancing accuracy. This approach offers a practical solution for assessing detectability under specified conditions. Other similar efforts with ANN models are detailed in (@ChapmanBird:2023, @Gerosa:2020, @Callister:2024).

In addition to providing trained ANN models for specific configurations, `gwsnr` offers users the flexibility to develop and train custom models tailored to their unique requirements. This adaptability allows for optimization based on variations in detector sensitivity, gravitational-wave properties, and other research-specific factors, ensuring maximum model effectiveness across different scenarios.

### Partial Scaling Interpolation Method with SNR Recalculation for Pdet Estimation

While the Partial Scaling method is highly efficient for aligned-spin systems, its utility can be further enhanced by recalculating the SNR for precessing systems within a predefined small range of generated SNRs. This is done by first obtaining optimal SNRs with the Partial Scaling method, selecting the SNRs near $\rho_{\rm th}$, and then recalculating the SNRs for these systems using the Noise-Weighted Inner Product Method. This approach allows us to leverage the speed of the Partial Scaling method while ensuring accurate SNR values for systems close to the detection threshold. The recalculated SNRs can then be used to estimate $P_{\rm det}$, providing a balance between computational efficiency and accuracy.

# Acknowledgements

The author gratefully acknowledges the substantial contributions from all who supported this research. Special thanks go to my academic advisors for their invaluable guidance and unwavering support. The interactions with my research colleagues have greatly enriched this work. The Department of Physics at The Chinese University of Hong Kong is acknowledged for the Postgraduate Studentship that made this research possible. Thanks are also due to the LIGO Laboratory for the computational resources, supported by National Science Foundation Grants No. PHY-0757058 and No. PHY-0823459.


<!-- **Interpolation Method**: Utilizes a 2D cubic spline technique (njit-ted) for the 'partialsnr' segment. -->

# References

<!-- # Mathematical Formulation

#### Modified FINDCHIRP Method: Partial Scaling Approach

The *`gwsnr`* package introduces the Partial Scaling method for SNR calculations in spin-less binary systems. This method, rooted in the FINDCHIRP algorithm (@Allen:2012), focuses on non-spinning inspiral-merger-ringdown (IMR) waveforms, in lalsimulation library (@lalsuite:2018), and particularly interpolates the Partial scaled SNR ($\rho_{1/2}$) based on mass parameters ($M,q$).

- **Interpolation Method**: Utilizes a 2D cubic spline technique (njit-ted) for the 'partialsnr' segment.

- **Equations**:

  - For a simple inspiral waveform, the optimal SNR is given by,
    $$\rho = F(D_l,\mathcal{M},\iota,\psi,\alpha, \delta) \sqrt{ 4\int_0^{f_{\rm LSO}} \frac{f^{-7/3}}{S_n(f)}df }$$

  - $F$ is defined as a function of luminosity distance ($D_l$), chirp mass ($\mathcal{M}$), inclination angle ($\iota$), polarization angles ($\psi$), right ascension ($\alpha$), and declination ($\delta$); refer to Eqn(D1) of @Allen:2012. $f$ is the frequency, $f_{\rm LSO}$ is the last stable orbit frequency and $S_n(f)$ is the detector's noise curve or power spectral density (psd).

  - Then, partial scaled SNR: $\rho_{1/2} = \sqrt{ 4\int_0^\infty \frac{f^{-7/3}}{S_n(f)}df } \approx \sqrt{ 4\int_0^{f_{\rm LSO}} \frac{f^{-7/3}}{S_n(f)}df }$

  - For an spinless frequency-domain IMR waveform with optimal SNR equal to $\rho$: $\rho_{1/2} = \rho\,/\, F(D_l,\mathcal{M},\iota,\psi,\alpha, \delta)$

  - $\rho_{1/2}$ is considered a function of $M$ and $q$.

#### Noise-Weighted Inner Product Method with Multiprocessing

This method is tailored for SNR calculations using frequency domain waveforms as defined in *`lalsimulation`* (@lalsuite:2018), including spin-precessing binary systems. `gwsnr` also supports JAX assited inner product, where the waveform generation is facilitated through the `ripple` package (@Edwards:2023). Key functions are optimized using `jax.jit` and parallelized with `jax.vmap`.

- **Methodology**: Combines waveform generation (multi-process), antenna pattern calculation (njit-ted), and noise-weighted inner product computation (njit-ted).

- **Equations**:

  - Inner product: $\left< a | b \right> = 4 \int_{f_{min}}^{f_{max}} \frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df$

  - Optimal SNR: $\rho = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_{\times}^2 \left< \tilde{h}_{\times} | \tilde{h}_{\times} \right> }$, for orthogonal $h_+$ and $h_{\times}$.

  - $h_{+\times}$ are frequency domain waveform polarizations, and $F_{+\times}$ are antenna patterns. 

These formulations highlight *`gwsnr`*'s capability to efficiently process diverse GW signals, enhancing data analysis accuracy and efficiency. 

#### Artificial Neural Network (ANN) Model for Pdet Estimation

The *`gwsnr`* package now incorporates an artificial neural network (ANN) model, developed using *`TensorFlow`* (@tensorflow:2015) and *`sklearn`* (@scikitlearn:2011), to rapidly estimate the Pdet in binary black hole (BBH) systems using the IMRPhenomXPHM waveform approximant. This complex IMR waveform model accounts for spin-precessing systems with subdominant harmonics. The ANN model is especially useful when precise signal-to-noise ratio (SNR) calculations are not critical, providing a quick and effective means of estimating Pdet. This value indicates detectability under Gaussian noise by determining if the SNR exceeds a certain threshold. Trained on a large dataset from the *`ler`* package, the ANN model uses 'partial scaled SNR' values as a primary input, reducing input dimensionality from 15 to 5 and enhancing accuracy. This approach offers a practical solution for assessing detectability under specified conditions. Other similar efforts with ANN models are detailed in @ChapmanBird:2023, @Gerosa:2020, @Callister:2024 etc.

In addition to providing trained ANN models for specific configurations, *`gwsnr`* offers users the flexibility to develop and train custom models tailored to their unique requirements. This adaptability allows for optimization based on variations in detector sensitivity, gravitational wave properties, and other research-specific factors, ensuring maximum model effectiveness across different scenarios. -->