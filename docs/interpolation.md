
# The Partial Scaling Interpolation Method

The `gwsnr` package implements the Partial Scaling method, an interpolation-assisted technique designed for the rapid calculation of SNR of binary systems with non-spinning or aligned-spin components. Although originally conceived for simple inspiral-only waveforms, the method has since been generalized to include full inspiral-merger-ringdown (IMR; frequency domain) waveforms available in the [lalsimulation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/modules.html) library. 

The core principle, adapted from the FINDCHIRP algorithm ([@Allen:2012](https://arxiv.org/pdf/gr-qc/0509116)), is to pre-compute and interpolate the most computationally expensive component of the SNR calculation—termed the "partial-SNR," denoted as $\rho_{\frac{1}{2}}$. This partial-SNR depends solely on the intrinsic parameters of the binary, such as total mass $M$, mass ratio $q$, and spin magnitudes $a_1$ and $a_2$. The final SNR is then quickly recovered through the application of a simple scaling factor. To further optimize performance, the entire procedure, including both antenna response computations and interpolation, is accelerated using either `numba.njit` compilation or `jax.jit` compilation and further parralelized using `jax.vmap`.

## Mathematical Formulation

The Partial Scaling method begins with the standard definition of the optimal SNR, $\rho$. For a simple inspiral waveform, following Eq.(2) of the Sec.([Noise-Weighted Inner Product Method](https://gwsnr.readthedocs.io/en/latest/innerproduct.html)) and Eq.(D1) of [@Allen:2012](https://arxiv.org/pdf/gr-qc/0509116), the SNR can be written as

$$
\begin{align}
\rho &= \sqrt{4 \int_{f_{\rm min}}^{f_{\rm max}} \frac{|\tilde{h}(f)|^2}{S_n(f)}\, df}, \notag \\
&= \left( \frac{1~\mathrm{Mpc}}{D_{\mathrm{eff}}} \right)
\sqrt{4 \mathcal{A}_{1~\mathrm{Mpc}}^2 ({\cal M})
\int_{f_{\rm min}}^{f_{\rm max}} \frac{f^{-7/3}}{S_n(f)}\, df }, \notag \\
&= \left(\frac{1~\mathrm{Mpc}}{D_{\mathrm{eff}}}\right) \mathcal{A}_{1~\mathrm{Mpc}} ({\cal M}) \sqrt{ 4\int_{f_{\rm min}}^{f_{\rm lso}} \frac{f^{-7/3}}{S_n(f)}df }, \tag{1}
\end{align}
$$

where $\tilde{h}(f)$ is the frequency-domain waveform, $S_n(f)$ is the detector noise power spectral density, and $D_{\text{eff}}$ is the effective distance to the source. The amplitude term, $\mathcal{A}_{1,\text{Mpc}}$, is a function of the chirp mass ${\cal M}$. Now, $D_{\text{eff}}$ and $\mathcal{A}_{1,\text{Mpc}}$ reads

$$
\begin{align}
\mathcal{A}_{\rm 1Mpc} &= \left(\frac{5}{24\pi}\right)^{\frac{1}{2}}
\left(\frac{GM_\odot/c^2}{1~\mathrm{Mpc}}\right)
\left(\frac{\pi GM_\odot}{c^3}\right)^{-1/6}
\left(\frac{\mathcal{M}}{M_\odot}\right)^{5/6} \,,\tag{2} \\
D_\mathrm{eff} &= D_l \left[
F_+^2 \left(\frac{1+\cos^2\iota}{2}\right)^2 +
F_\times^2 \cos^2\iota
\right]^{-\frac{1}{2}}.\tag{3} \\
\end{align}
$$

$\mathcal{M}$ is the chirp mass, and in terms of ($M,q$) it reads $\mathcal{M} = M \left( \frac{q}{1+q^2} \right)^{3/5}$, where $M = m_1 + m_2$ is the total mass and $q = m_2/m_1$ is the mass ratio. 
$D_l$ is the luminosity distance, $\iota$ is the inclination angle, and $F_+$ and $F_\times$ are the antenna patterns of the detector.
The integral runs from a minimum frequency $f_{\min}$ (typically $20$ Hz) up to the frequency of the last stable orbit, which is determined by the total mass of the binary system as $f_{\rm LSO} = 1/(6^{3/2} \pi) (G\,M/c^3)^{-\frac{1}{2}}$.

A key insight is to isolate the computationally intensive integral, which is independent of extrinsic parameters such as distance, inclination angle, and sky location. This term is defined as the partial-SNR, $\rho_{\frac{1}{2}}$ and it reads

$$
\begin{align}
\rho_{\frac{1}{2}} &= \left(\frac{D_\mathrm{eff}}{1~\mathrm{Mpc}}\right) \mathcal{M}^{-5/6} \times \rho\,, \tag{4}\\
&= \left(\frac{5}{24\pi}\right)^{\frac{1}{2}}
\left(\frac{GM_\odot/c^2}{1~\mathrm{Mpc}}\right)
\left(\frac{\pi GM_\odot}{c^3}\right)^{-1/6}
\left(\frac{1}{M_\odot}\right)^{5/6}
\sqrt{4 \int_{20}^{f_{\rm LSO}} \frac{f^{-7/3}}{S_n(f)}\, df} \tag{5}
\end{align}
$$

For more general IMR waveforms the same principle applies. The full SNR, $\rho_{\text{bilby}}$, is first calculated using a library such as `bilby`, and the partial-SNR is analogously defined as

$$
\begin{align}
\rho_{\frac{1}{2}} &= \left(\frac{D_\mathrm{eff}}{1~\mathrm{Mpc}}\right) \mathcal{M}^{-5/6} \times \rho_{\rm bilby}\,, \tag{6}
\end{align}
$$

## Interpolation Procedure

To obtain the SNR for a new set of binary parameters, the method interpolates $\rho_{\frac{1}{2}}$ from a pre-computed grid. 
For non-spinning IMR waveforms, a two-dimensional grid in $(M, q)$ is constructed, with grid spacing optimized to minimize interpolation error: the $q$ axis is logarithmically spaced, while the $M$ axis is inverse-logarithmically spaced. Given a new parameter set $(M_{\text{new}}, q_{\text{new}})$, [nested one-dimensional cubic spline interpolation](https://gwsnr.readthedocs.io/en/latest/interpolation.html#nested-1d-cubic-spline-interpolation) is used to determine the corresponding value of $\rho_{\frac{1}{2}}$ from the pre-computed grid.

For spin-aligned IMR waveforms, the approach is extended to a four-dimensional grid spanning $(M, q, a_1, a_2)$, where $a_1$ and $a_2$ are the dimensionless spin magnitudes. The $(M, q)$ axes maintain the optimized spacing, while the spin axes are uniformly spaced within a user-defined range.

Once the interpolated value of $\rho_{\frac{1}{2}}$ is obtained, the final optimal SNR is simply recovered by scaling with the chirp mass and effective distance:

$$
\begin{align}
\rho = \rho_{\frac{1}{2}} \times \left( \frac{D_{\text{eff}}}{1~\mathrm{Mpc}} \right) \left( \frac{{\cal M}}{M_\odot} \right)^{5/6}.\tag{7}
\end{align}
$$

This methodology facilitates rapid and precise SNR evaluation throughout the entire parameter space relevant to gravitational wave detection. The implementation achieves high computational efficiency by employing just-in-time (JIT) compilation via `numba.njit`, or alternatively, by utilizing `jax.jit` in conjunction with vectorization through `jax.vmap`. This dual-backend design enables flexible deployment, supporting both multi-threaded parallelization on CPUs and GPU acceleration. As a result, large-scale SNR computations can be conducted with minimal computational overhead, making the approach well suited for demanding gravitational wave data analysis applications.

## Nested 1D Cubic Spline Interpolation

The Partial Scaling method relies on nested one-dimensional (1D) cubic spline interpolation to efficiently estimate the partial-SNR, for a new set of binary parameters. This hierarchical approach avoids the complexity of multi-dimensional spline fitting by applying a sequence of 1D interpolations across each parameter dimension.

To illustrate the process, consider the 2D case for a non-spinning system with a pre-computed grid of $\rho_{\frac{1}{2}}$ values over $M$ and $q$ parameter space. The interpolation proceeds in several key steps. First, the axis arrays for $M$ and $q$ are obtained, and the precomputed $\rho_{\frac{1}{2}}$ values are loaded from a pickle file. These $\rho_{\frac{1}{2}}$ values are defined for each $(M, q)$ pair on the grid.

Given a new set of binary parameters $(M_{\rm new}, q_{\rm new})$, the next step is to identify the four nearest neighboring points along each axis: for $M$, the points $[M_{i-1}, M_i, M_{i+1}, M_{i+2}]$ are selected such that $M_i \leq M_{\rm new} \leq M_{i+1}$; similarly, for $q$, the points $[q_{j-1}, q_j, q_{j+1}, q_{j+2}]$ are chosen so that $q_j \leq q_{\rm new} \leq q_{j+1}$.

For each fixed value of $q$, a one-dimensional cubic spline interpolation is performed along the $M$ axis to estimate the values of $\rho_{\frac{1}{2}}$ at $M_{\rm new}$. This process yields a set of four intermediate values, corresponding to $q_{j-1}$, $q_j$, $q_{j+1}$, and $q_{j+2}$:

$$
\left[ \rho_{\frac{1}{2}}(q_{j-1}, M_{\rm new}),\, \rho_{\frac{1}{2}}(q_j, M_{\rm new}),\, \rho_{\frac{1}{2}}(q_{j+1}, M_{\rm new}),\, \rho_{\frac{1}{2}}(q_{j+2}, M_{\rm new}) \right].
$$

A subsequent cubic spline interpolation is then carried out along the $q$ axis, using these values, to determine the final interpolated result $\rho_{\frac{1}{2}}(M_{\rm new}, q_{\rm new})$.

<p align="center">
    <img src="_static/grid2D.jpg" alt="2D Grid Interpolation" width="400"/>
</p>

For the spin-aligned IMR waveform, the methodology is conceptually identical but operates in a four-dimensional grid spanning $(M, q, a_1, a_2)$. The interpolation follows a hierarchical scheme, beginning with the $a_1$ and $a_2$ axes, followed by $M$, and finally $q$, recursively applying the one-dimensional cubic spline interpolation at each stage until the final value, $\rho_{\frac{1}{2}}(M_{\rm new}, q_{\rm new}, a_{1, \rm new}, a_{2, \rm new})$, is obtained.

In the presence of edge cases—particularly when the target parameter lies near or beyond the boundaries of the precomputed grid—the methodology employs a robust fallback: the nearest available value of $\rho_{\frac{1}{2}}$ is utilized. Specifically, if $M_{\rm new} < M_1$ (or $M_0$), where $M_0$ and $M_1$ represent the first and second points along the $M$ axis, the scheme resorts to simple linear interpolation (or extrapolation) based on these initial grid points. This approach ensures both the stability and consistency of the interpolation procedure, even at or outside the defined limits of the parameter space.

## Why interpolation is possible?

Interpolation is feasible for frequency-domain IMR waveforms restricted to the dominant (2,2)-mode because the partial-SNR exhibits smooth and continuous dependence on the binary's intrinsic parameters. As demonstrated in the figure below, the partial-SNR varies smoothly with total mass $M$, mass ratio $q$, and spin magnitudes $a_1$ and $a_2$. This continuous behavior enables accurate interpolation across the parameter space defined by the pre-computed grid. Importantly, for this class of waveforms, the partial-SNR depends only on intrinsic parameters and remains independent of extrinsic parameters such as luminosity distance, inclination angle, and sky location.

<p align="center">
    <img src="_static/gwsnr_partial_snr_examples.png" alt="Partial SNR Parameter Dependencies" width="500"/>
</p>
<p align="center">
    <em>Figure: Partial SNR ρ<sub>1/2</sub> as a function of intrinsic parameters for aligned-spin IMRPhenomD waveforms. The four panels demonstrate the smooth variation of ρ<sub>1/2</sub> with respect to total mass M, mass ratio q, and spin magnitudes a<sub>1</sub> and a<sub>2</sub>, with other parameters held constant. The continuous and monotonic behavior enables accurate interpolation across the parameter space.</em>
</p>

Spin-precessing systems, with higher harmonics, the relationship is not as smooth, and interpolation may not yield accurate results. In such cases, the noise-weighted inner product method is recommended.

## Catmull-Rom Spline Interpolation

To estimate the partial SNR, $\rho_{\frac{1}{2}}$, for arbitrary binary parameters, `gwsnr` employs [Catmull-Rom spline interpolation](https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline). This technique is particularly well-suited for interpolating smoothly between pre-computed grid values, offering both continuity and accuracy with less computational cost than standard cubic [spline interpolation](https://en.wikipedia.org/wiki/Spline_interpolation).

Consider interpolation along the $M$-axis while keeping $q$, $a_1$, and $a_2$ fixed. To interpolate the partial SNR, $\rho_{\frac{1}{2}}$, at an arbitrary mass $M_{\text{new}}$, we use four neighboring grid points: $(M_0, \rho_{\frac{1}{2},0})$, $(M_1, \rho_{\frac{1}{2},1})$, $(M_2, \rho_{\frac{1}{2},2})$, and $(M_3, \rho_{\frac{1}{2},3})$, where $M_1 \leq M_{\text{new}} \leq M_2$.

The cubic interpolant is written as:

$$
\rho_{\frac{1}{2}}(M_{\rm new}) = a_0 + a_1 t + a_2 t^2 + a_3 t^3
$$

where $t$ is the normalized coordinate:

$$
t = \frac{M_{\rm new} - M_1}{M_2 - M_1}, \quad \text{with}\quad 0 \leq t \leq 1
$$

so that $M_1$ maps to $t=0$ and $M_2$ maps to $t=1$.

To determine the coefficients, we require:

$$
\begin{cases}
\rho_{\frac{1}{2}}(M_1) = a_0 = \rho_{\frac{1}{2},1} \\
\rho_{\frac{1}{2}}(M_2) = a_0 + a_1 + a_2 + a_3 = \rho_{\frac{1}{2},2} \\
\rho'_{\frac{1}{2}}(M_1) = \frac{a_1}{h} = \frac{\rho_{\frac{1}{2},2} - \rho_{\frac{1}{2},0}}{2h} \\
\rho'_{\frac{1}{2}}(M_2) = \frac{a_1 + 2a_2 + 3a_3}{h} = \frac{\rho_{\frac{1}{2},3} - \rho_{\frac{1}{2},1}}{2h}
\end{cases}
$$

where $h = M_2 - M_1$ is the grid spacing between the two interior points.

These conditions encode:

* The interpolant passes through the central two points.
* The derivatives at the boundaries are set by the Catmull-Rom prescription (using the values at the next neighbors).

Solving for $a_0, a_1, a_2, a_3$ in terms of $\rho_{\frac{1}{2},i}$ and $h$ gives:

$$
\begin{align*}
a_0 &= \rho_{\frac{1}{2},1} \\
a_1 &= \frac{1}{2} (\rho_{\frac{1}{2},2} - \rho_{\frac{1}{2},0}) \\
a_2 &= \rho_{\frac{1}{2},0} - 2.5\rho_{\frac{1}{2},1} + 2\rho_{\frac{1}{2},2} - 0.5\rho_{\frac{1}{2},3} \\
a_3 &= \frac{1}{2} (\rho_{\frac{1}{2},3} - 3\rho_{\frac{1}{2},2} + 3\rho_{\frac{1}{2},1} - \rho_{\frac{1}{2},0}) \\
\end{align*}
$$

with $t$ as defined above.

This approach ensures a smooth and physically consistent interpolation for SNR values across the parameter space, without the computational expense of a full cubic spline solution which requires inverting a matrix of size $8 \times 8$.

## Example Usage

```python
# loading GWSNR class from the gwsnr package
from gwsnr import GWSNR
import numpy as np

# initializing the GWSNR class with default configuration and interpolation method
# for aligned-spinning IMRPhenomD waveform
gwsnr_aligned_spins = GWSNR(
    npool=8,
    snr_type='interpolation_aligned_spins', # or 'interpolation_aligned_spins_jax'
    gwsnr_verbose=False,
)

nsamples = 50000
mtot = np.random.uniform(2*4.98, 2*112.5,nsamples)
mass_ratio = np.random.uniform(0.2,1,size=nsamples)
param_dict = dict(
    # convert to component masses
    mass_1 = mtot / (1 + mass_ratio), # in solar masses
    mass_2 = mtot * mass_ratio / (1 + mass_ratio), # in solar masses
    luminosity_distance = np.random.uniform(10,10000, size=nsamples), # in Mpc
    theta_jn = np.random.uniform(0,2*np.pi, size=nsamples), # inclination angle in radians
    ra = np.random.uniform(0,2*np.pi, size=nsamples),  # right ascension in radians
    dec = np.random.uniform(-np.pi/2,np.pi/2, size=nsamples),  # declination in radians
    psi = np.random.uniform(0,2*np.pi, size=nsamples), # polarization angle in radians
    phase = np.random.uniform(0,2*np.pi, size=nsamples), # coalescence phase in radians
    geocent_time = 1246527224.169434*np.ones(nsamples), # geocentric time in GPS seconds
    a_1 = np.random.uniform(-0.8,0.8, size=nsamples), # dimensionless spin of the primary black hole
    a_2 = np.random.uniform(-0.8,0.8, size=nsamples), # dimensionless spin of the secondary black hole
)

interp_snr_aligned_spins = gwsnr_aligned_spins.snr(gw_param_dict=param_dict) # or gwsnr_aligned_spins.snr(**param_dict)
```

## Accuracy and Performance

The accuracy of the Partial Scaling interpolation method is generally very high, with deviations typically less than 0.5% (and absolute errors on the order of 0.01) compared to the noise-weighted inner product method. This level of precision is sufficient for most gravitational wave data analysis applications, particularly when considering the inherent uncertainties in astrophysical parameters.

<p align="center">
    <img src="_static/snr_comparison_interpolation.png" alt="Partial SNR Parameter Dependencies" width="100%"/>
</p>
<p align="center">
    <em>Figure: Comparison of the predicted SNR using the Partial Scaling method (interpolation) against the true SNR from the Bilby library for aligned-spin IMRPhenomD waveforms. The close agreement demonstrates the effectiveness of the interpolation approach.</em>
</p>
