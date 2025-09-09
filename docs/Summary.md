# Summary

<figure align="center">
  <img src="_static/snr.png" alt="snr" width="100%" style="margin: 0; padding: 0;"/>
  <figcaption align="left">
    <b>Figure.</b> Detection of a gravitational wave signal in noisy data. Top panel: Signal-to-noise ratio (SNR) as a function of GPS time, with the SNR threshold (orange dashed line) indicating the minimum required for confident detection. The sharp peak crossing the threshold marks a detectable GW event. Bottom panel: Detector strain data versus GPS time, showing the gravitational wave waveform template (pink) aligned with a signal embedded in background noise (black).
  </figcaption>
</figure>


Gravitational waves (GWs)—ripples in spacetime predicted by Einstein’s theory of General Relativity—have revolutionized astrophysics since their first direct detection in 2015. These signals, emitted by the mergers of compact objects such as binary black holes (BBHs), binary neutron stars (BNSs), and neutron star-black hole pairs (NSBHs), provide unique insights into the universe. A central quantity in GW data analysis is the **signal-to-noise ratio** (SNR), which quantifies the strength of a GW signal relative to the noise in detectors like LIGO, Virgo, and KAGRA. Reliable SNR estimation is essential for confirming GW detections and performing astrophysical inference. However, modern GW research—especially in population simulations and hierarchical Bayesian inference with selection effects—requires the computation of SNRs for vast numbers of systems, making traditional methods based on noise-weighted inner products prohibitively slow.

The `gwsnr` Python package addresses this computational bottleneck, offering a flexible, high-performance, and user-friendly framework for SNR and probability of detection ($P_{\rm det}$) estimation. At its core, `gwsnr` leverages [NumPy](https://numpy.org/) vectorization along with Just-In-Time (JIT) compilation via [Numba](https://numba.pydata.org/), [JAX](https://github.com/google/jax) and [MLX](https://ml-explore.github.io/mlx/build/html/index.html), as well as Python multiprocessing, to deliver exceptional performance.

`gwsnr` should not be confused with other software packages use in the GW detection pipelines, e.g `PyCBC`, `GstLAL` etc. Our software is specifically designed for rapid SNR computation in large-scale astrophysical simulations.

### Key Features

- **Noise-Weighted Inner Product with Multiprocessing**: Provides accurate SNR calculations for arbitrary frequency-domain waveforms, including those with spin precession and higher-order harmonics available in [lalsimulation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/modules.html). The method is enhanced with python multiprocessing across multiple CPUs and JIT compiled functions, with optional support for JAX-based waveform libraries like [ripple](https://github.com/tedwards2412/ripple).  

- **Partial Scaling Interpolation**: An innovative and highly efficient interpolation method for accurately calculating the SNR of non-precessing (spinless or aligned-spin) binary systems. This approach dramatically reduces computation time, making large-scale analyses practical.

- **ANN-Based $P_{\rm det}$ Estimation**: Employs a trained Artificial Neural Network (ANN) to provide fast probability of detection ($P_{\rm det}$) estimates via SNR calculations for precessing BBH systems. This feature is especially valuable when rapid detection assessments are needed without requiring precise SNR values.

- **Hybrid SNR Recalculation**: A balanced approach that combines the speed of the partial scaling method (or ANN-based estimation) with the precision of the noise-weighted inner product, ensuring high accuracy for systems near the detection threshold.

- **Horizon Distance Calculation**: Implements both analytical and numerical methods to compute the horizon distance for gravitational wave sources, allowing users to assess detector sensitivity and detection capabilities across various configurations.

- **Integration and Flexibility**: Offers a user-friendly interface to combine various detector noise models, waveform models, detector configurations, and signal parameters.

These capabilities make `gwsnr` an invaluable tool for GW data analysis, particularly for determining the rates of lensed and unlensed GW events (as demonstrated by its use in the [ler](https://ler.readthedocs.io/en/latest/) package and related works), and for modeling selection biases in hierarchical Bayesian frameworks.

Full mathematical and implementation details are provided in the Sections: [Inner Product](https://gwsnr.readthedocs.io/en/latest/innerproduct.html), [Interpolation](https://gwsnr.readthedocs.io/en/latest/interpolation.html), [ANN](https://gwsnr.readthedocs.io/en/latest/ann.html), [Hybrid](https://gwsnr.readthedocs.io/en/latest/hybrid.html).


