# Summary

<p align="center">
    <img src="_static/snr.png" alt="Your Logo" width="50%" style="margin: 0; padding: 0;">
</p>
<p align="center">
    <em>Detection of a gravitational wave signal in noisy data. Top (cartoon): Signal-to-noise ratio (SNR) as a function of GPS time, with the SNR threshold (orange dashed line) indicating the minimum required for confident detection. The sharp peak crossing the threshold marks a detectable GW event. Bottom (cartoon): Strain versus GPS time, showing the gravitational wave signal (blue) embedded in background noise (grey).</em>
</p>

Gravitational waves (GWs)—ripples in spacetime predicted by Einstein’s theory of General Relativity—have revolutionized astrophysics since their first direct detection in 2015 [1,2]. These signals, emitted by the mergers of compact objects such as binary black holes (BBHs), binary neutron stars (BNSs), and black hole–neutron star pairs, provide unique insights into the universe. A central quantity in GW data analysis is the **signal-to-noise ratio** (SNR), which quantifies the strength of a GW signal relative to the noise in detectors like LIGO[3], Virgo[4], and KAGRA[5]. Reliable SNR estimation is essential for confirming GW detections and performing astrophysical inference [6]. However, modern GW research—especially in population simulations [7] and hierarchical Bayesian inference with selection effects [8]—requires the computation of SNRs for vast numbers of systems, making traditional methods based on noise-weighted inner products prohibitively slow [9,10].

The **`gwsnr`** Python package addresses this computational bottleneck, offering a flexible, high-performance, and user-friendly framework for SNR and probability of detection ($P_{\rm det}$) estimation. At its core, `gwsnr` leverages [NumPy](https://numpy.org/) vectorization along with Just-In-Time (JIT) compilation via [Numba](https://numba.pydata.org/) and [JAX](https://github.com/google/jax) to deliver exceptional performance.

### Key Features

- **Noise-Weighted Inner Product with Multiprocessing**: Provides accurate SNR calculations for arbitrary waveforms, including those with spin precession and higher-order harmonics available in [lalsimulation][11]. The method is enhanced with multiprocessing and JIT compilation to accelerate computation, with optional support for JAX-based waveform libraries like [ripple][12].
- **Partial Scaling Interpolation**: An innovative and highly efficient interpolation method for accurately calculating the SNR of non-precessing (spinless or aligned-spin) binary systems. This approach dramatically reduces computation time, making large-scale analyses practical.
- **ANN-Based $P_{\rm det}$ Estimation**: Incorporates a trained Artificial Neural Network (ANN) for rapid probability of detection estimates for precessing BBH systems. This is particularly useful when exact SNR values are not required, and follows similar approaches adopted in recent literature [13–15].
- **Hybrid SNR Recalculation**: A balanced approach that combines the speed of the partial scaling method with the precision of the noise-weighted inner product, ensuring high accuracy for systems near the detection threshold.
- **Integration and Flexibility**: Offers a user-friendly interface to combine various detector noise models, waveform models, detector configurations, and signal parameters.

These capabilities make `gwsnr` an invaluable tool for GW data analysis, particularly for determining the rates of lensed and unlensed GW events (as demonstrated by its use in the [ler][16] package and related works [17–22]), and for modeling selection biases in hierarchical Bayesian frameworks [8].

### Mathematical Formulation

`gwsnr` implements two primary methods for SNR computation and two for $P_{\rm det}$ estimation:

- **Noise-Weighted Inner Product Method**: Computes the optimal SNR $\rho$ using the noise-weighted inner product [23]. For a signal with plus ($h_+$) and cross ($h_\times$) polarizations, the SNR is given by:
    $$
    \rho = \sqrt{ F_+^2 \left< \tilde{h}_+ | \tilde{h}_+ \right> + F_{\times}^2 \left< \tilde{h}_{\times} | \tilde{h}_{\times} \right> }
    $$
    where $F_+$ and $F_\times$ are the detector's antenna patterns and the inner product is:
    $$
    \left< a | b \right> = 4 \Re \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}(f)\tilde{b}^*(f)}{S_n(f)} df
    $$

- **Partial Scaling Interpolation Method**: For non-precessing systems, this method uses a pre-computed grid of "partial-scaled SNR" ($\rho_{1/2}$) values, which depend only on intrinsic parameters like mass and spin. The SNR for a new set of parameters is found by scaling the interpolated grid value:
    $$
    \rho = \rho_{1/2} \times \left(\frac{\mathcal{M}}{M_\odot}\right)^{5/6} \times \left(\frac{1~\mathrm{Mpc}}{D_\mathrm{eff}}\right)
    $$
    where $\mathcal{M}$ is the chirp mass and $D_{\text{eff}}$ is the effective distance.

- **Hybrid & ANN Methods for $P_{\rm det}$**: For precessing binaries where the above interpolation is inexact, `gwsnr` offers two solutions. The hybrid method uses the fast interpolation to identify candidate systems near the detection threshold ($\rho_{\rm th}=8$) and then recalculates their SNRs accurately using the inner product. Alternatively, a trained ANN (developed with [TensorFlow](https://www.tensorflow.org/) and [scikit-learn](https://scikit-learn.org/)) provides direct $P_{\rm det}$ estimates, enabling rapid assessments of detectability.

Full mathematical and implementation details are provided in the [package documentation](https://gwsnr.readthedocs.io/en/latest/).

---

## References

1. B.P. Abbott et al., Phys. Rev. Lett. 116, 061102 (2016), [doi](http://dx.doi.org/10.1103/PhysRevLett.116.061102)  
2. B.P. Abbott et al., Phys. Rev. Lett. 116, 241102 (2016), [doi](http://dx.doi.org/10.1103/physrevlett.116.241102)  
3. J. Aasi et al. (LIGO Scientific Collaboration), Class. Quantum Grav. 32, 074001 (2015), [doi](https://dx.doi.org/10.1088/0264-9381/32/7/074001)  
4. F. Acernese et al. (Virgo Collaboration), Class. Quantum Grav. 32, 024001 (2015), [doi](https://dx.doi.org/10.1088/0264-9381/32/2/024001)  
5. T. Akutsu et al. (KAGRA Collaboration), arXiv:2005.05574 (2020), [arXiv](https://arxiv.org/abs/2005.05574)  
6. B.P. Abbott et al., Phys. Rev. D 93, 122003 (2016), [doi](http://dx.doi.org/10.1103/physrevd.93.122003)  
7. B.P. Abbott et al., Astrophys. J. Lett. 818, L22 (2016), [doi](http://dx.doi.org/10.3847/2041-8205/818/2/l22)  
8. E. Thrane & C. Talbot, Publ. Astron. Soc. Aust. 36, e010 (2019), [doi](http://dx.doi.org/10.1017/pasa.2019.2)  
9. S. Taylor et al., Phys. Rev. D 98, 083017 (2018), [doi](http://dx.doi.org/10.1103/physrevd.98.083017)  
10. D. Gerosa et al., Phys. Rev. D 102, 103020 (2020), [doi](http://dx.doi.org/10.1103/physrevd.102.103020)  
11. LIGO Scientific Collaboration, [LALSuite software](https://doi.org/10.7935/GT1W-FZ16)  
12. E. Edwards et al., Phys. Rev. D 110, 064028 (2023), [doi](https://doi.org/10.1103/PhysRevD.110.064028)  
13. B. Chapman & R. Bird, MNRAS 522, 1847 (2023), [doi](http://dx.doi.org/10.1093/mnras/stad1397)  
14. D. Gerosa et al., Phys. Rev. D 102, 103020 (2020), [doi](http://dx.doi.org/10.1103/physrevd.102.103020)  
15. T. Callister et al., Phys. Rev. D 110, 123041 (2024), [doi](http://dx.doi.org/10.1103/PhysRevD.110.123041)  
16. [ler package documentation](https://arxiv.org/abs/2407.07526)  
17. L. Leo et al., arXiv:2403.16532 (2024), [arXiv](https://arxiv.org/abs/2403.16532)  
18. A. More et al., arXiv:2502.02536 (2025), [arXiv](https://arxiv.org/abs/2502.02536)  
19. H. Janquart et al., MNRAS 524, 1099 (2023), [doi](https://doi.org/10.1093/mnras/stad2909)  
20. B.P. Abbott et al., Astrophys. J. 923, 262 (2021), [doi](http://dx.doi.org/10.3847/1538-4357/ac23db)  
21. LIGO-Virgo-KAGRA Collaboration, arXiv:2304.08393 (2023), [arXiv](https://arxiv.org/abs/2304.08393)  
22. J. Wierda et al., Astrophys. J. 922, 223 (2021), [doi](https://doi.org/10.3847/1538-4357/ac1bb4)  
23. B. Allen et al., Phys. Rev. D 85, 122006 (2012), [doi](http://dx.doi.org/10.1103/physrevd.85.122006)  
