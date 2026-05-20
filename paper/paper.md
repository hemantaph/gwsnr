---
title: '$gwsnr$: A Python package for efficient signal-to-noise calculation of gravitational waves'
tags:
  - Python
  - astrophysics
  - statistics
  - GWs
  - LIGO
authors:
  - name: Hemantakumar Phurailatpam
    orcid: 0000-0002-0471-3724
    affiliation: "1"
  - name: Otto Akseli HANNUKSELA
    orcid: 0000-0002-3887-7137
    affiliation: "1"
affiliations:
  - name: Department of Physics, The Chinese University of Hong Kong, Shatin, New Territories, Hong Kong
    index: 1
date: 12 January 2024
bibliography: paper.bib
---

# Summary

Gravitational waves (GWs), ripples in spacetime predicted by Einstein's theory of General Relativity, have revolutionized astrophysics since their first detection in 2015. Emitted by cataclysmic events such as mergers of binary black holes (BBHs), binary neutron stars (BNSs), and black hole-neutron star pairs (BH-NSs), these waves provide a unique window into the cosmos. A central quantity in GW analysis is the Signal-to-Noise Ratio (SNR), which measures the strength of a GW signal relative to the background noise in detectors such as LIGO [@LIGO:2015, @Abbott:2020, @Buikema:2020], Virgo [@VIRGO:2015, @VIRGO:2019], and KAGRA [@Akutsu:2020, @Aso:2013]. While real detections are established using a False-Alarm Rate (FAR) threshold, under stationary Gaussian noise assumptions the condition that the SNR exceeds a chosen threshold can serve as a practical proxy [@Essick:2023, @Essick:2024]. This proxy is especially useful in simulations of detectable events and in studies aimed at extracting astrophysical information [@Abbott:2016:detection]. 

To support these large-scale studies, we introduce $gwsnr$, a Python package designed to provide efficient and flexible resources for computing the SNR and estimating the probability of detection for large simulated populations of GW events. By accelerating these core calculations, $gwsnr$ makes massive population-level astrophysical simulations computationally viable.

# Statement of need

Applications such as population simulations for rate estimation [@Abbott:2016:rates] and hierarchical Bayesian inference with selection effects [@Thrane:2019, @Essick:2024] require repeated and efficient computation of the probability of detection ($P_{\rm det}$). This probability is generally derived from the SNR. However, traditional approaches that rely on noise-weighted inner products for SNR evaluation are computationally demanding and often impractical for large-scale analyses [@Taylor:2018, @Gerosa:2020]. 

The $gwsnr$ package is specifically designed to solve this computational bottleneck. It provides fast evaluation of the [optimal-SNR](https://gwsnr.hemantaph.com/detectionstatistics.html#defining-optimal-snr) ($\rho_{\rm opt}$) and $P_{\rm det}$. The target audience includes LIGO-Virgo-KAGRA (LVK) researchers and astrophysicists working on population studies, selection-effect modeling, and astrophysical rate estimation. For massive simulations, $gwsnr$ estimates $P_{\rm det}$ by evaluating $\rho_{\rm opt}$ against a detection-statistic threshold under flexible detector, waveform, and population settings. It additionally allows researchers to statistically model the (noise-realised) [observed/matched-filter SNR](https://gwsnr.hemantaph.com/detectionstatistics.html#defining-match-filter-snr) ($\rho_{\rm obs}$) based on $\rho_{\rm opt}$ under stationary Gaussian noise assumption. The package accelerates $\rho_{\rm opt}$ evaluation using specialized numerical methods, making rapid calculations possible under diverse astrophysical configurations that would otherwise be computationally prohibitive. Although intended for a variety of research purposes, $gwsnr$ operates effectively as a specialized detectability calculator and is heavily utilized by the `ler` [@ler:2024] software for simulating detectable strongly lensed GW events and calculating their rates [@Leo:2024, @More:2025].

# State of the field

In GW data analysis, commonly used packages such as `Bilby` [@Ashton:2019], `PyCBC` [@Usman:2016], and `GstLAL` [@Cannon:2021], operating alongside `LALSuite` [@lalsuite:2018], provide high-precision tools for Bayesian inference, matched-filter searches, signal processing, and waveform generation. While these packages are widely adopted, they are not primarily designed for the fast and repeated evaluation of $\rho_{\rm opt}$ and $P_{\rm det}$ across millions of simulated events. Direct noise-weighted inner-product calculations using these standard tools become a significant computational bottleneck. To address this, related studies have developed machine-learning methods to estimate detectability. Gerosa et al. [@Gerosa:2020] trained classifiers using optimal-SNR thresholds, while Callister et al. [@Callister:2024] trained an emulator directly from search-pipeline detections defined by FAR criteria. Similarly, Chapman-Bird et al. [@ChapmanBird:2023] implemented an SNR-based selection model for extreme mass ratio inspirals using neural networks. While these machine-learning methods provide rapid evaluation, their applicability depends strictly on specific training data, detector configurations, and waveform models. Furthermore, errors are heavily concentrated near the sharp detection boundary where minor variations can misclassify marginal events [@Gerosa:2020]. Modifying detector noise curves or waveform families in these emulators requires computationally expensive dataset regeneration and complete model retraining.

The $gwsnr$ package addresses these limitations by providing a modular framework dedicated strictly to efficient detectability calculations. The build-versus-contribute justification is founded on the necessity for an adaptable library focused solely on repeated SNR and $P_{\rm det}$ evaluations with user-controlled parameters. Unlike broad inference packages or rigid machine-learning emulators, $gwsnr$ explicitly computes $\rho_{\rm opt}$ through accelerated numerical pathways while allowing $\rho_{\rm obs}$ to be modeled statistically. The detection threshold can also be estimated from injection catalogues using empirical sensitivity information [@Essick:2023]. This framework allows researchers to seamlessly switch between partial-scaling interpolation, artificial neural network approximations, and full multiprocessing inner-product integration depending on their specific accuracy and speed requirements. By interfacing with established C-based waveform infrastructure such as `LALSuite` and JAX-based tools such as `ripplegw` [@Edwards:2023], $gwsnr$ fills a unique scholarly role as an adaptable high-speed detectability engine for broader simulation pipelines.

# Software design

The architecture of $gwsnr$ is driven by the trade-off between accuracy, speed, and waveform generality. Providing a single evaluation method is insufficient for diverse astrophysical studies. Therefore, $gwsnr$ implements multiple calculation paths under a unified interface parameterized by the detector power spectral density (PSD), antenna response, waveform model, detector network, source parameters, and detection threshold.

The baseline calculation executes the standard frequency-domain [inner product](https://gwsnr.hemantaph.com/innerproduct.html#noise-weighted-inner-product-method) [@Allen:2012] given by the equation

$$
\langle a | b \rangle =
4 \Re \int_{f_{\min}}^{f_{\max}}
\frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df
$$

where $S_n(f)$ is the detector PSD. The optimal SNR is $\rho=\sqrt{\langle h|h\rangle}$. For the two polarizations $h_+$ and $h_\times$ the optimal SNR is calculated as

$$
\rho =
\sqrt{
F_+^2 \langle \tilde{h}_+|\tilde{h}_+\rangle
+
F_\times^2 \langle \tilde{h}_\times|\tilde{h}_\times\rangle
}.
$$

This path is prioritized for its generality and supports spin-precessing systems alongside waveforms with subdominant modes. $gwsnr$ accelerates this conventional approach using Python multiprocessing, Just-In-Time (JIT) compilation via `numba.njit` [@numba:2022] for antenna-pattern generation, and optional `JAX` [@jax:2018] backends integrated with `ripplegw` [@Edwards:2023]. With $n$ allocated processes, [multiprocessing delivers speedups](https://gwsnr.hemantaph.com/performancesummary.html#inner-product-performance) approaching a factor of $n$, with minimal overhead.

For non-spinning and aligned-spin binaries, $gwsnr$ implements a fundamentally faster [partial-scaling interpolation](https://gwsnr.hemantaph.com/interpolation.html#the-partial-scaling-interpolation-method) method based on the FINDCHIRP scaling [@Allen:2012]. This architectural choice avoids repeated inner-product integrations by isolating the computationally expensive mass-dependent and spin-dependent components. The method precomputes the partial-scaled SNR defined as

$$
\rho_{1/2}
=
\frac{D_\mathrm{eff}}{\mathcal{M}^{5/6}}
\rho_{\rm opt}
$$

where $\mathcal{M}$ is the chirp mass and $D_{\rm eff}$ is the effective distance. The quantity $\rho_{1/2}$ is stored on multidimensional irregular grids using local cubic-Hermite splines. New SNR values are rapidly recovered by interpolation and rescaling using the equation

$$
\rho_{\rm opt}
=
\rho_{1/2}
\frac{\mathcal{M}^{5/6}}{D_\mathrm{eff}} .
$$

The full $\rho_{\rm opt}$ generation is wrapped in JIT-compiled functions for parallel execution on CPUs (via the `numba` backend) and on supported GPUs, including Apple Silicon (via the `MLX` backend [@mlx:2023]) and Nvidia GPUs (via the `JAX` backend). For randomly sampled parameters, this method achieves accuracy exceeding $99.5\%$ compared with traditional noise-weighted inner-product calculations using `Bilby` [@Ashton:2019], and processes up to one million $\rho_{\rm opt}$ values in $\sim 200$ milliseconds (max case) on GPU backends, with [speedups](https://gwsnr.hemantaph.com/performancesummary.html#interpolation-performance) of $\sim 5{,}000\times$ relative to `Bilby`. The interpolation grid must be generated once beforehand ($\lesssim 1$ min with default settings) and is stored in JSON files for reuse; `numba.njit` compilation overhead is negligible thereafter.

As a supplementary path, $gwsnr$ includes [artificial neural network (ANN) estimation](https://gwsnr.hemantaph.com/ann.html#ann-based-pdet-estimation) using `tensorflow` [@tensorflow:2015] and `scikit-learn` [@scikitlearn:2011]. This design is applied to complex waveform settings where direct partial-scaling interpolation is geometrically impractical. The model strategically uses partial-scaled SNR quantities to reduce the input dimensionality from fifteen parameters down to five. Users are also provided with the tools to train their own models for different detector, waveform, and population settings.

To manage the boundary classification errors inherent to approximate methods, $gwsnr$ introduces a [hybrid SNR recalculation](https://gwsnr.hemantaph.com/hybrid.html#hybrid-strategy-for-spin-precessing-systems) workflow. It estimates initial SNRs using partial scaling or ANN prediction, isolates events near the detection threshold $\rho_{\rm th}$, and exactly recalculates those specific systems using the direct inner-product method. This design preserves the speed of interpolation while guaranteeing reliability near the critical detection boundary where small SNR errors can directly affect $P_{\rm det}$.

The [probability of detection](https://gwsnr.hemantaph.com/probabilityofdetection.html#probability-of-detection-calculation) is evaluated by applying a threshold to either $\rho_{\rm opt}$ or $\rho_{\rm obs}$, denoted by $\rho_{\rm opt,th}$ and $\rho_{\rm obs,th}$, respectively. For a deterministic threshold on a chosen SNR quantity $\rho$,

$$
P_{\rm det}
=
P({\rm det}\mid\vec{\theta})
=
\Theta(\rho-\rho_{\rm th}),
$$

where $\vec{\theta}$ denotes the GW parameters and $\Theta$ is the Heaviside step function. If $\rho_{\rm obs}$ is treated as a noise-dependent random variable, the probability is averaged over noise realizations,

$$
P({\rm det}\mid\vec{\theta})
=
P(\rho_{\rm obs}>\rho_{\rm obs,th}\mid\vec{\theta})
=
1-F_{\rho_{\rm obs}}(\rho_{\rm obs,th}\mid\vec{\theta}),
$$

where $F_{\rho_{\rm obs}}$ is the cumulative distribution function under the assumed noise model. Under the stationary Gaussian noise approximation, `gwsnr` supports two models for $\rho_{\rm obs}$. The first treats $\rho_{\rm obs}$ as a normal random variable centred on $\rho_{\rm opt}$ with unit variance [@Fishbach:2020; @Abbott:2019]. The second treats $\rho_{\rm obs}^2$ as a non-central $\chi^2$ variable. Using the convention of `scipy.stats.ncx2` [@scipy:2020],

$$
\rho_{\rm obs}^2
\sim
\chi^2_{\rm nc}
\left(
k=2,
\lambda=\rho_{\rm opt}^2
\right)
$$

for a single detector, and

$$
\rho_{\rm obs,net}^2
\sim
\chi^2_{\rm nc}
\left(
k=2N,
\lambda=\rho_{\rm opt,net}^2
\right)
$$

for a network of $N$ detectors.

`gwsnr` also supports user-defined thresholds and [provides tools](https://gwsnr.hemantaph.com/examples/threshold.html#SNR-Threshold-Finder-Example) for estimating thresholds from injection catalogues, following Essick [@Essick:2023]. The current implementation treats this threshold as parameter-independent. Parameter-dependent thresholds and the corresponding $P_{\rm det}$ calculation are left for future development.

Finally, $gwsnr$ calculates the [horizon distance](https://gwsnr.hemantaph.com/horizondistance.html#horizon-distance) ($D_{\rm hor}$), representing the maximum distance an optimally oriented source can be detected [@Allen:2012]. The analytical path simply rescales a known effective distance using

$$
D_{\rm hor}
=
\frac{\rho_{\rm opt}}{\rho_{\rm opt,th}} D_{\rm eff}.
$$

The alternative numerical method maximizes the SNR over sky location and solves for the luminosity distance $d_L$ where

$$
\rho(d_L) - \rho_{\rm opt,th} = 0 .
$$

# Research impact statement

The $gwsnr$ package provides the computational speed and accuracy required for detectability calculations within large simulated populations of compact binary mergers. By making massive SNR evaluations computationally viable, the software actively supports astrophysical [rate estimation](https://ler.hemantaph.com/examples/LeR_custom_functions.html), [detector-sensitivity](https://gwsnr.hemantaph.com/examples/horizon_distance.html) studies, and [selection-effect modeling](https://ler.hemantaph.com/examples/selection_function.html) in hierarchical Bayesian inference. A primary realized impact of $gwsnr$ is its functional integration as the core detectability calculator within the `ler` software package [@ler:2024]. In this capacity, it identifies detectable unlensed and strongly lensed gravitational-wave events to calculate their expected occurrence rates, directly facilitating population-level lensing analyses in published literature [@Janquart:2023, @Leo:2024, @More:2025].

The community readiness of $gwsnr$ is demonstrated by its formal peer review within the LIGO-Virgo-KAGRA Scientific Collaboration [@gwsnrlvkpnpreview:2024]. The software maintains active support through [GitHub issues](https://github.com/hemantaph/gwsnr/issues) and [collaboration communication channels](https://chat.ligo.org/). The package is publicly accessible on the Python Package Index where it has recorded over 400 downloads [@gwsnrpypi:2026]. Comprehensive documentation provides the underlying theory, tutorials, and benchmarks necessary to reproduce all core calculations [@gwsnrdocs:2026]. These factors provide specific and compelling evidence that $gwsnr$ successfully supports independent simulation pipelines tailored with custom detector, waveform, and population parameters.

# AI usage disclosure

No generative AI tools were used to develop the software, write this manuscript, or prepare the accompanying materials.

# Acknowledgements

Hemantakumar Phurailatpam acknowledges the Department of Physics at The Chinese University of Hong Kong for the Postgraduate Studentship that facilitated this research. Hemantakumar Phurailatpam and Otto A. Hannuksela acknowledge support from the Research Grants Council of Hong Kong, Project Nos. CUHK 14304622 and 14307923, the start-up grant from The Chinese University of Hong Kong, and the Direct Grant for Research from the Research Committee of The Chinese University of Hong Kong. The authors also thank the LIGO Laboratory for computational resources, supported by National Science Foundation Grants No. PHY-0757058 and No. PHY-0823459.

# References