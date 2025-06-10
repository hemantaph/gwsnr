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

Gravitational waves (GWs), ripples in spacetime predicted by Einstein's theory of General Relativity, have revolutionized astrophysics since their first detection in 2015 (@Abbott:2016). These waves are emitted by cataclysmic events like the merging of binary black holes (BBHs), binary neutron stars (BNSs) and BH-NS pairs, providing a unique window into the cosmos. A critical aspect of GW analysis is the Signal-to-Noise Ratio (SNR). SNR quantifies the strength of a GW signal relative to the background noise in a detector, like LIGO (@LIGO2015, @Abbott2020, @PhysRevD.102.062003), Virgo (@VIRGO2015, @virgo:2019) or KAGRA (@akutsu2020overviewkagradetectordesign, @PhysRevD:88:043007). This ratio is pivotal in confirming the detection of GWs and extracting astrophysical information from them (@Abbott:2016:detection). However, specific scenarios in GW research, particularly in simulations of detectable GW events (@Abbott:2016:rates) and in hierarchical Bayesian analysis (@Thrane:2019) where selection effects are considered, demand extensive and efficient computation of SNR. This requirement presents a significant challenge, as conventional computational approaches, such as noise-weighted inner product, are typically time-consuming and impractical for such specialized and large-scale analyses (@Taylor:2018, @Gerosa:2020).

# Statement of Need

The *`gwsnr`* Python package addresses the need for efficient SNR computation in GW research. It provides a flexible and user-friendly interface, allowing users to combine various detector noise models, waveform models, detector configurations, and signal parameters. *`gwsnr`* enhances SNR calculations through several key features. Firstly, it utilizes an innovative interpolation method, employing a partial-scaling approach for accurately interpolating the SNR of GWs from spin-less binary systems. Secondly, the package features a noise-weighted inner product method, similar to that in the *`bilby`* package (@bilby), but enhanced with multiprocessing capabilities. This parallel processing is crucial for handling large datasets and computationally intensive analyses. Thirdly, a trained Artificial Neural Network (ANN) model is incorporated for rapid 'probability of detection' (Pdet) estimation for BBH systems with spin precession. Lastly, *`gwsnr`* leverages the *`numba`*'s Just-In-Time (njit) compiler (@numba), which optimizes performance by compiling Python code into machine code at runtime, drastically reducing execution times. This combination of advanced techniques and user-friendly design makes gwsnr a valuable tool for GW data analysis, particularly in simulating detectable compact binary mergers, determining rates of both lensed and unlensed GW events (as demonstrated by its use in the *`ler`* package; @ler, @ng2024), and will help in the analysis of selection effects within hierarchical Bayesian frameworks (@Thrane:2019).

# Mathematical Formulation

#### Modified FINDCHIRP Method: Partial Scaling Approach

The *`gwsnr`* package introduces the Partial Scaling method for SNR calculations in spin-less binary systems. This method, rooted in the FINDCHIRP algorithm (@Allen:2012), focuses on non-spinning inspiral-merger-ringdown (IMR) waveforms, in lalsimulation library (@lalsuite), and particularly interpolates the Partial scaled SNR ($\rho_{1/2}$) based on mass parameters ($M,q$).

- **Interpolation Method**: Utilizes a 2D cubic spline technique (njit-ted) for the 'partialsnr' segment.

- **Equations**:

  - For a simple inspiral waveform, the optimal SNR is given by,
    $$\rho = F(D_l,\mathcal{M},\iota,\psi,\alpha, \delta) \sqrt{ 4\int_0^{f_{ISCO}} \frac{f^{-7/3}}{S_n(f)}df }$$

  - $F$ is defined as a function of luminosity distance ($D_l$), chirp mass ($\mathcal{M}$), inclination angle ($\iota$), polarization angles ($\psi$), right ascension ($\alpha$), and declination ($\delta$); refer to Eqn(D1) of @Allen:2012. $f$ is the frequency, $f_{ISCO}$ is the last stable orbit frequency and $S_n(f)$ is the detector's noise curve or power spectral density (psd).

  - Then, partial scaled SNR: $\rho_{1/2} = \sqrt{ 4\int_0^\infty \frac{f^{-7/3}}{S_n(f)}df } \approx \sqrt{ 4\int_0^{f_{ISCO}} \frac{f^{-7/3}}{S_n(f)}df }$

  - For an spinless frequency-domain IMR waveform with optimal SNR equal to $\rho$: $\rho_{1/2} = \rho\,/\, F(D_l,\mathcal{M},\iota,\psi,\alpha, \delta)$

  - $\rho_{1/2}$ is considered a function of $M$ and $q$.

#### Noise-Weighted Inner Product Method with Multiprocessing

This method is tailored for SNR calculations using frequency domain waveforms as defined in *`lalsimulation`* (@lalsuite), including spin-precessing binary systems. `gwsnr` also supports JAX assited inner product, where the waveform generation is facilitated through the `ripple` package (@Edwards:2023sak). Key functions are optimized using `jax.jit` and parallelized with `jax.vmap`.

- **Methodology**: Combines waveform generation (multi-process), antenna pattern calculation (njit-ted), and noise-weighted inner product computation (njit-ted).

- **Equations**:

  - Inner product: $\left< a | b \right> = 4 \int_{f_{min}}^{f_{max}} \frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df$

  - Optimal SNR: $\rho = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_{\times}^2 \left< \tilde{h}_{\times} | \tilde{h}_{\times} \right> }$, for orthogonal $h_+$ and $h_{\times}$.

  - $h_{+\times}$ are frequency domain waveform polarizations, and $F_{+\times}$ are antenna patterns. 

These formulations highlight *`gwsnr`*'s capability to efficiently process diverse GW signals, enhancing data analysis accuracy and efficiency. 

#### Artificial Neural Network (ANN) Model for Pdet Estimation

The *`gwsnr`* package now incorporates an artificial neural network (ANN) model, developed using *`TensorFlow`* (@tensorflow2015) and *`sklearn`* (@scikitlearn), to rapidly estimate the Pdet in binary black hole (BBH) systems using the IMRPhenomXPHM waveform approximant. This complex IMR waveform model accounts for spin-precessing systems with subdominant harmonics. The ANN model is especially useful when precise signal-to-noise ratio (SNR) calculations are not critical, providing a quick and effective means of estimating Pdet. This value indicates detectability under Gaussian noise by determining if the SNR exceeds a certain threshold. Trained on a large dataset from the *`ler`* package, the ANN model uses 'partial scaled SNR' values as a primary input, reducing input dimensionality from 15 to 5 and enhancing accuracy. This approach offers a practical solution for assessing detectability under specified conditions. Other similar efforts with ANN models are detailed in @ChapmanBird2023, @Gerosa:2020, @Callister2024 etc.

In addition to providing trained ANN models for specific configurations, *`gwsnr`* offers users the flexibility to develop and train custom models tailored to their unique requirements. This adaptability allows for optimization based on variations in detector sensitivity, gravitational wave properties, and other research-specific factors, ensuring maximum model effectiveness across different scenarios.

# Acknowledgements

The author gratefully acknowledges the substantial contributions from all who supported this research. Special thanks go to my academic advisors for their invaluable guidance and unwavering support. The interactions with my research colleagues have greatly enriched this work. The Department of Physics at The Chinese University of Hong Kong is acknowledged for the Postgraduate Studentship that made this research possible. Thanks are also due to the LIGO Laboratory for the computational resources, supported by National Science Foundation Grants No. PHY-0757058 and No. PHY-0823459.

## III. Interpolation with Partial Scaling method

The `gwsnr` package introduces the Partial Scaling method for SNR calculations in spin-less and spin-aligned binary systems using interpolation assisted technique. 
The idea for 'Partial Scaling method' is initially conceived for spin-less inspiral only waveforms, but it can be extended to IMR waveform (spinless and spin-aligned) systems as well.
This method, rooted in the FINDCHIRP algorithm ([@Allen:2012](https://arxiv.org/pdf/gr-qc/0509116)), focuses on non-spinning and aligned-spins frequency domain waveforms, in `lalsimulation` library (@lalsuite), and particularly interpolates the Partial scaled SNR ($\rho_{1/2}$) based on mass parameters ($M,q$) and spin-magnitude ($a_1, a_2$) (if aligned-spins).

Performance is optimized by though `JAX`'s Just-In-Time (`jit`) compilation, which accelerates both the antenna response functions calculation and the interpolation process. 

### Mathematical Formulation

Partial Scaling method, can be formulated, for spinless inspiral-only system with interpolation wrt to mass parameters, total-mass $M$ and mass-ratio $q$, as follows:
For a simple **inspiral** waveform, following Eq.(2) of the Sec.([II](#ii-noise-weighted-inner-product-method)) and Eq.(D1) of [@Allen:2012](https://arxiv.org/pdf/gr-qc/0509116), the optimal SNR is given by,
$$
\begin{align}
\rho &= \sqrt{4 \int_{f_{\rm min}}^{f_{\rm max}} \frac{|\tilde{h}(f)|^2}{S_n(f)}\, df}, \notag \\
&= \left( \frac{1~\mathrm{Mpc}}{D_{\mathrm{eff}}} \right)
\sqrt{4 \mathcal{A}_{1~\mathrm{Mpc}}^2 ({\cal M})
\int_{f_{\rm min}}^{f_{\rm max}} \frac{f^{-7/3}}{S_n(f)}\, df }, \notag \\
&= F(D_{\mathrm{eff}}, {\cal M}) \sqrt{ 4\int_{f_{\rm min}}^{f_{\rm lso}} \frac{f^{-7/3}}{S_n(f)}df }, \tag{B1}
\end{align}
$$
where
$$
\begin{align}
F(D_{\mathrm{eff}}, {\cal M}) &= \left(\frac{1~\mathrm{Mpc}}{D_{\mathrm{eff}}}\right) \mathcal{A}_{1~\mathrm{Mpc}} ({\cal M})\,, \tag{B2}\\
\mathcal{A}_{\rm 1Mpc} &= \left(\frac{5}{24\pi}\right)^{1/2}
\left(\frac{GM_\odot/c^2}{1~\mathrm{Mpc}}\right)
\left(\frac{\pi GM_\odot}{c^3}\right)^{-1/6}
\left(\frac{\mathcal{M}}{M_\odot}\right)^{5/6} \,,\tag{B3} \\
D_\mathrm{eff} &= D \left[
    F_+^2 \left(\frac{1+\cos^2\iota}{2}\right)^2 +
    F_\times^2 \cos^2\iota
\right]^{-1/2}\tag{B4} \\
\end{align}
$$

$\mathcal{M}$ is the chirp mass, and in terms of ($M,q$) it reads $\mathcal{M} = M \left( \frac{q}{1+q^2} \right)^{3/5}$, where $M = m_1 + m_2$ is the total mass and $q = m_2/m_1$ is the mass ratio. 
We take $f_{\rm min} = 20$ Hz and $f_{\rm max} = f_{\rm LSO}$, the last stable orbit frequency, which is determined by the total mass of the binary system as $f_{ISCO} = 1/(6^{3/2} \pi) (G\,M/c^3)^{-1/2}$.

The idea is to separate out, from the inner product equation, the computationally expensive integral part (which is independedent of extrinsic parameter and detector configuration) and interpolate it with respect to ($M,q$). This part, which we term as 'Partial-SNR', reads:
$$
\begin{align}
\rho_{1/2} &= \left(\frac{D_\mathrm{eff}}{1~\mathrm{Mpc}}\right) \mathcal{M}^{-5/6} \times \rho\,, \tag{B5}\\
&= \left(\frac{5}{24\pi}\right)^{1/2}
\left(\frac{GM_\odot/c^2}{1~\mathrm{Mpc}}\right)
\left(\frac{\pi GM_\odot}{c^3}\right)^{-1/6}
\left(\frac{1}{M_\odot}\right)^{5/6}
\sqrt{4 \int_{20}^{f_{\rm LSO}} \frac{f^{-7/3}}{S_n(f)}\, df} \tag{B6}
\end{align}
$$

For general (spin-aligned or spin-less) IMR waveform, we calculate $\rho$ using `bilby` (now $\rho_{\rm bilby}$), and follow the same idea as above and Eq.(B4) still holds, and now Partial-SNR reads:
$$
\begin{align}
\rho_{1/2} &= \left(\frac{D_\mathrm{eff}}{1~\mathrm{Mpc}}\right) \mathcal{M}^{-5/6} \times \rho_{\rm bilby}\,, \tag{B7}
\end{align}
$$

For spin-less IMR waveform, we make a 2D grid of $(M,q)$ and calculate $\rho_{1/2}$ for each point in the grid and store it in a pickle file. The number of elements in the each axis doesn't need to be same and can be set by the user. Also the elements in each axis are not equally space, $q$ axis is logarithmically spaced, while $M$ axis is inverse-logarithmically spaced, to minimize error with less number of points. But the value in each axis should be strictly increasing. To generate the $\rho_{1/2}$ for a newly provided set of parameters $(M_{\rm new},q_{\rm new})$, we load the pickle file, and then we use nested 1D cubic spline interpolation (see Sec.([IV]())), instead of bicubic spline interpolation, using the grid and the corresponding precomuted values of $\rho_{1/2}$

For spin-aligned IMR waveform, we carry out similar interpolation but in in 4D space of $(M,q,a_1,a_2)$. The number of elements and spacing for $(M,q)$ axes are same as above, while $a_1$ and $a_2$ axes are uniformly spaced in a specified range $[-a_{\rm max}, a_{\rm min}]$. 

## IV. Nested 1D Cubic Spline Interpolation

Let's take the example of Partial-SNR interpolation for spin-less IMR waveform, where we have a 2D grid of $(M,q)$ and corresponding $\rho_{1/2}$ values. The interpolation is done in the following steps:

- **Step 1**: Get the precomputed grid of $(M,q)$ and load the corresponding $\rho_{1/2}$ values from a pickle file.
- **Step 2**: For a new set of parameters $(M_{\rm new},q_{\rm new})$, find the four nearest points in axes [$M_{i-1}, M_i, M_{i+1}, M_{i+2}$] and [$q_{j-1}, q_j, q_{j+1}, q_{j+2}$] and $M_i\leq M_{\rm new} \leq M_{i+1}$, such that $q_j \leq q_{\rm new} \leq q_{j+1}$. 
- **Step 3**: For each fixed $q_j$, we perform a 1D cubic spline interpolation in the $M$ axis, using the four nearest points [$M_{i-1}, M_i, M_{i+1}, M_{i+2}$] and their corresponding $\rho_{1/2}$ values of $\right[(q_{j-1}, M_{\rm new}), (q_j, M_{\rm new}), (q_{j+1}, M_{\rm new}), (q_{j+2}, M_{\rm new})\left]$. This gives us a new interpolated value $\rho_{1/2}(M_{\rm new}, q_j)$.


If the new point(s) are 


<!-- **Interpolation Method**: Utilizes a 2D cubic spline technique (njit-ted) for the 'partialsnr' segment. -->

# References