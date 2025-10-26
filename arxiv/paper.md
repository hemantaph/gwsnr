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

Gravitational waves (GWs), ripples in spacetime predicted by Einstein's theory of General Relativity, have revolutionized astrophysics since their first detection in 2015. Emitted by cataclysmic events such as mergers of binary black holes (BBHs), binary neutron stars (BNSs), and black hole-neutron star pairs (BH-NSs), these waves provide a unique window into the cosmos. 

A central quantity in GW analysis is the Signal-to-Noise Ratio (SNR), which measures the strength of a GW signal relative to the background noise in detectors such as LIGO (@LIGO:2015, @Abbott:2020, @Buikema:2020), Virgo (@VIRGO:2015, @VIRGO:2019), and KAGRA (@Akutsu:2020, @Aso:2013). While real detections are established using a False-Alarm Rate (FAR) threshold, under stationary Gaussian noise assumptions the condition that the SNR exceeds a chosen threshold can serve as a practical proxy (@Essick:2023, @Essick:2024), especially in simulations of detectable events and in studies aimed at extracting astrophysical information (@Abbott:2016:detection).

Applications such as population simulations for rate estimation (@Abbott:2016:rates) and hierarchical Bayesian inference with selection effects (@Thrane:2019, @Essick:2024) require repeated and efficient computation of the Probability of Detection ($P_{\rm det}$), which is generally derived from SNR. However, traditional approaches that rely on noise-weighted inner products for SNR evaluation are computationally demanding and often impractical for such large-scale analyses (@Taylor:2018, @Gerosa:2020).

# Statement of Need

The *`gwsnr`* Python package addresses this challenge by providing efficient and flexible tools for computing the optimal SNR ($\rho_{\rm opt}$). This quantity depends on the intrinsic and extrinsic source parameters, the detector antenna response ($F_{+,\times}$), and the noise power spectral density (PSD) (@Allen:2012). The primary use case of $\rho_{\rm opt}$ in *`gwsnr`* is the estimation of $P_{\rm det}$, which is evaluated against a detection statistics threshold.

The package provides a flexible and user-friendly interface for combining detector noise models, waveform families, detector configurations, and signal parameters. It accelerates $\rho_{\rm opt}$ evaluation using a **partial-scaling interpolation** method for non-precessing binaries and a multiprocessing **inner-product** routine for frequency-domain waveforms implemented in `lalsuite` (@lalsuite:2018), including those with spin precession and subdominant modes. For rapid $P_{\rm det}$ estimation, *`gwsnr`* also supports ANN-based models and a Hybrid SNR recalculation scheme. Finally, using an optimal-SNR threshold $\rho_{\rm opt,thr}$, the package computes the horizon distance ($D_{\rm hor}$), a standard measure of detector sensitivity, via both analytical (@Allen:2012) and numerical methods.   

High performance is achieved through *`NumPy`* vectorization (@numpy:2022) and Just-in-Time (JIT) compilation with *`Numba`* (@numba:2022), with optional GPU acceleration available via *`JAX`* (@jax:2018) and *`MLX`* (@mlx:2023). These JIT compilers translate Python code into optimized machine code at runtime, while built-in parallelization strategies such as `numba.prange`, `jax.vmap`, and `mlx.vmap` maximize efficiency on both CPUs and GPUs.  

This combination of efficiency and usability makes *`gwsnr`* a valuable tool for GW data analysis. It enables large-scale simulations of compact binary mergers, facilitates the estimation of detectable lensed and unlensed event rates (as demonstrated in the *`ler`* package; @ler:2024, @Leo:2024, @More:2025, @Janquart:2023, @Abbott:2021, @ligolensing:2023, @Wierda:2021, @Wempe:2022), and supports the treatment of selection effects through $P_{\rm det}$ in hierarchical Bayesian frameworks (@Thrane:2019, @Essick:2023).


# Mathematical Formulation and Methods Overview

Following are the key mathematical formulations and methods implemented in *`gwsnr`* for SNR calculation, $P_{\rm det}$ estimation, and $D_{\rm hor}$ computation.

### Noise-Weighted Inner Product  

The standard frequency-domain inner product (@Allen:2012) between two signals $\tilde{a}(f)$ and $\tilde{b}(f)$ is  

$$
\langle a | b \rangle = 4 \Re \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df ,
$$  

where $S_n(f)$ is the detector PSD. The optimal SNR is $\rho = \sqrt{\langle h|h\rangle}$, and for polarizations $h_+, h_\times$:  

$$
\rho = \sqrt{F_+^2 \langle \tilde{h}_+|\tilde{h}_+\rangle + F_\times^2 \langle \tilde{h}_\times|\tilde{h}_\times\rangle} .
$$  

Although waveform generation is costly, *`gwsnr`* accelerates it using `multiprocessing`, `numba.njit`, and optional `jax` backends (with `ripple`; @Edwards:2023).

### Partial Scaling Interpolation  

For aligned-spin or non-spinning binaries, *`gwsnr`* adapts FINDCHIRP (@Allen:2012) to precompute a partial-scaled SNR,  

$$
\rho_{1/2} = \frac{D_\mathrm{eff}}{\mathcal{M}^{5/6}} \rho_{\rm opt} ,
$$  

where $\mathcal{M}$ is the chirp mass and $D_{\rm eff}$ the effective distance. $\rho_{1/2}$ is stored on a parameter grid (2D for non-spinning, 4D for aligned spins). New SNRs are recovered by spline interpolation and rescaling:  

$$
\rho = \rho_{1/2}\, \frac{\mathcal{M}^{5/6}}{D_\mathrm{eff}} .
$$  

This replaces costly integrations with interpolation, enabling major speed-ups.  

### ANN-based $P_{\rm det}$ Estimation  

*`gwsnr`* includes an ANN built with `tensorflow` (@tensorflow:2015) and `scikit-learn` (@scikitlearn:2011), trained to approximate $\rho_{\rm opt}$ for BBH systems with the IMRPhenomXPHM waveform, which includes spin precession and subdominant modes. While the ANN is poor at estimating $\rho_{\rm opt}$ directly, its outputs are effective for $P_{\rm det}$, since detectability depends on threshold crossing rather than precise values.  

Trained on large *`ler`* datasets, the model uses partial-scaled SNRs to reduce input dimensionality (15 to 5) and accelerate detectability estimates under stationary Gaussian noise. Users can also retrain the ANN for different detectors or astrophysical settings. Related work includes (@ChapmanBird:2023, @Gerosa:2020, @Callister:2024).  

### Hybrid SNR Recalculation for $P_{\rm det}$ Estimation  

The Partial Scaling method is efficient for aligned-spin systems but unreliable for precessing binaries, and the ANN-based approach is less accurate. To address this, *`gwsnr`* uses a hybrid strategy: it first estimates SNRs with Partial Scaling or ANN, identifies signals near the threshold $\rho_{\rm th}$, and then recalculates them with the Noise-Weighted Inner Product.  

This approach retains the speed of approximations while ensuring accuracy for systems close to the detection limit, producing more reliable $P_{\rm det}$ estimates.  

### Statistical Models for $P_{\rm det}$  

In *`gwsnr`*, estimation of $P_{\rm det}$ is based on a detection threshold for the observed (matched-filter) SNR, $\rho_{\rm obs,thr}$. The observed SNR, $\rho_{\rm obs}$, is modeled either as a Gaussian random variate centered at $\rho_{\rm opt}$ (or $\rho_{\rm opt,net}$ for a detector network) with unit variance (@Fishbach:2020, @Abbott:2019), or as a non-central $\chi$ distribution (`scipy.stats.ncx2`; @scipy:2020) with non-centrality parameter $\lambda = \rho_{\rm opt}$ (or $\rho_{\rm opt,net}$) and two degrees of freedom for a single detector, extended to $2N$ for a network of $N$ detectors (@Essick:2023).  

*`gwsnr`* uses precomputed $\rho_{\rm obs,thr}$ values derived from semianalytic sensitivity estimates of GW transient injection catalogues (following @Essick:2023). The package also supports custom threshold computation from user-provided catalogue data, including parameter-dependent thresholds that vary with intrinsic properties such as total observed mass ($m_{\rm tot, obs}$).

### Horizon Distance Calculation  

$D_{\rm hor}$ is a standard measure of detector sensitivity, defined as the maximum distance at which an optimally oriented source can be detected with a given threshold $\rho_{\rm opt,thr}$ (@Allen:2012). *`gwsnr`* computes $D_{\rm hor}$ using two methods. 

The **analytical method** rescales a known $D_{\rm eff}$ by  

$$
D_{\rm hor} = \frac{\rho_{\rm opt}}{\rho_{\rm th}} D_{\rm eff}.
$$  

The **numerical method** maximizes SNR over sky location, then solves for the luminosity distance ($d_L$) where 

$$
\rho(d_L) - \rho_{\rm opt, thr} = 0 .
$$  

# Acknowledgements

The author gratefully acknowledges the substantial contributions from all who supported this research. Special thanks go to my academic advisors for their invaluable guidance and unwavering support. The interactions with my research colleagues have greatly enriched this work. The Department of Physics at The Chinese University of Hong Kong is acknowledged for the Postgraduate Studentship that made this research possible. Thanks are also due to the LIGO Laboratory for the computational resources, supported by National Science Foundation Grants No. PHY-0757058 and No. PHY-0823459.


# References