# Probability of Detection Calculation

The `gwsnr` package provides tools to calculate the probability of detecting a gravitational-wave (GW) signal, denoted as $P_{\rm det}$. This calculation is based on whether the signal’s signal-to-noise ratio (SNR) is strong enough to be distinguished from the detector’s background noise and deemed significant. For a given detection threshold, $\rho_{\rm th}$, `gwsnr` offers two ways to model this probability.


## 1. Deterministic Probability (Step Function)

The most straightforward approach treats detection as a binary outcome based on the [optimal SNR](innerproduct.md#optimal-snr-calculation), $\rho_{\rm opt}$. If the signal’s intrinsic strength is greater than the threshold, it is considered detected. This relationship is modeled as a step function:

$$
P_{\rm det}(\theta) =
\begin{cases}
1 & \text{if } \rho_{\rm opt}(\theta) > \rho_{\rm th} \\
0 & \text{otherwise}
\end{cases}\tag{1}
$$

Here, $\theta$ represents the set of parameters defining the GW source. This method is computationally simple and provides a clear “yes” or “no” answer for detectability based on the ideal SNR.

---

## 2. Probabilistic Detection (Gaussian Noise Model)

A more realistic model acknowledges that detector noise causes the measured matched-filter SNR, $\rho_{\rm mf}$, to fluctuate around the true optimal SNR, $\rho_{\rm opt}$. Assuming the background noise is Gaussian, these fluctuations can be described by a normal distribution with a mean equal to the optimal SNR and a standard deviation of one.

Following the convention in GW astronomy (e.g., [Thrane & Talbot 2019](https://arxiv.org/abs/1809.02293)), the probability density of measuring a specific $\rho_{\rm mf}$ for a signal with parameters $\theta$ is:

$$
p(\rho_{\rm mf} \mid \theta) = \frac{1}{\sqrt{2\pi}} \exp\left[ -\frac{1}{2}(\rho_{\rm mf} - \rho_{\rm opt}(\theta))^2 \right]\tag{2}
$$

The probability of detection, $P_{\rm det}$, is the probability that the measured SNR will exceed the threshold $\rho_{\rm th}$. This is given by the integral:

$$
P_{\rm det}(\theta) = \int_{\rho_{\rm th}}^{\infty} p(\rho_{\rm mf} \mid \theta) \, d\rho_{\rm mf}
= \int_{\rho_{\rm th}}^{\infty} \frac{1}{\sqrt{2\pi}} \exp\left[ -\frac{1}{2}(x - \rho_{\rm opt}(\theta))^2 \right] dx \tag{3}
$$

In `gwsnr`, this integral is efficiently calculated using the cumulative distribution function (CDF) of the standard normal distribution.

## Implementation

It is important to clarify that `gwsnr` does not perform a separate matched-filter analysis to find $\rho_{\rm mf}$, as it is not required to get $P_{\rm det}$ evident from Eqn.(3). Instead, it first calculates the optimal SNR ($\rho_{\rm opt}$) using one of its highly efficient methods (such as Partial Scaling or the noise-weighted inner product). This $\rho_{\rm opt}$ value is then used as the mean in the Gaussian model above to derive a realistic detection probability. The implementation in `gwsnr` is equivalent to:

```python
from scipy.stats import norm

# Probability of detection: 1 minus the CDF of (threshold - optimal SNR)
P_det = 1 - norm.cdf(snr_th - snr_opt)
```

## Example Usage

```python
# loading GWSNR class from the gwsnr package
from gwsnr import GWSNR
import numpy as np
# initializing the GWSNR class with inner product as the signal-to-noise ratio type
gwsnr = GWSNR()

# defining the parameters for the gravitational wave signal for a BBH with GW150914 like parameters
param_dict= dict(
    mass_1=np.array([36.0]), # mass of the primary black hole in solar masses
    mass_2=np.array([29.0]), # mass of the secondary black hole in solar masses
    luminosity_distance=np.array([440.0]), # luminosity distance in Mpc
    theta_jn=np.array([1.0]), # inclination angle in radians
    ra=np.array([3.435]), # right ascension in radians
    dec=np.array([-0.408]), # declination in radians
    psi=np.array([0.0]),  # polarization angle in radians
    geocent_time=np.array([1126259462.4]), # geocentric time in GPS seconds
    a_1=np.array([0.3]), # dimensionless spin of the primary black hole
    a_2=np.array([0.2]), # dimensionless spin of the secondary black hole
    tilt_1=np.array([0.5]), # tilt angle of the primary black hole in radians
    tilt_2=np.array([0.8]), # tilt angle of the secondary black hole in radians
    phi_12=np.array([0.0]), # Relative angle between the primary and secondary spin of the binary in radians
    phi_jl=np.array([0.0]), # Angle between the total angular momentum and the orbital angular momentum in radians
)

# signal-to-noise ratio with detectors LIGO-Hanford, LIGO-Livingston, and Virgo with O4 observing run sensitivity
snr_dict = gwsnr.snr(**param_dict)

# probability of detection with a threshold of 8. The result is a dict with keys as detector names and values as the detection probabilities, and also the network detection probability.

# 1. Deterministic Probability (Step Function)
pdet_bool = gwsnr.probability_of_detection(snr_dict=snr_dict, snr_th=8, snr_th_net=8, type="bool") 

# 2. Probabilistic Detection (Gaussian Noise Model)
pdet_gaussian = gwsnr.probability_of_detection(snr_dict=snr_dict, snr_th=8, snr_th_net=8, type="matched_filter")
```      

<!-- ## Probability of Detection Calculation

The `gwsnr` package provides tools to evaluate the probability of detecting a GW signal, denoted as $P_{\rm det}$. The calculation is based on whether the observed SNR exceeds a specified threshold, $\rho_{\rm th}$, for either individual detectors or a detector network. For most practical applications with Gaussian noise, using an SNR threshold is a reliable (proxy) criterion for detection.

Within gwsnr, two principal approaches are available for computing $P_{\rm det}$: one based on the optimal SNR, $\rho_{\rm opt}$, and another that considers the statistical nature of the matched-filter SNR, $\rho_{\rm mf}$.

## Detection Probability Using the Optimal SNR

When using the optimal SNR, the detection criterion is straightforward. A signal is considered detected if $\rho_{\rm opt}$ surpasses the threshold $\rho_{\rm th}$. In this case, the detection probability is represented as a step function:

$$
P_{\rm det} = P(\theta\mid \rho_{\rm opt}) =
\begin{cases}
1 & \text{if } \rho_{\rm opt} > \rho_{\rm th}, \\
0 & \text{otherwise},
\end{cases}
$$

where $\theta$ represents the set of parameters for the GW signal.

## Detection Probability with Matched-Filter SNR

For a more realistic scenario, the matched-filter SNR, $\rho_{\rm mf}$, fluctuates due to noise and follows a normal distribution with mean $\rho_{\rm opt}(\theta)$ and unit variance for a given set of parameters $\theta$. Following [Thrane et al. 2019](https://arxiv.org/abs/1809.02293), the probability density for measuring a particular value $\rho_{\rm mf}$ is

$$
p(\rho_{\rm mf} | \theta) = \frac{1}{\sqrt{2\pi}} \exp\left[-\frac{1}{2} \left( \rho_{\rm mf} - \rho_{\rm opt}(\theta) \right)^2 \right].
$$

The probability that the measured SNR exceeds the threshold, i.e., the probability of detection, is then given by

$$
P_{\rm det} = P(\theta\mid \rho_{\rm opt}) = \int_{\rho_{\rm th}}^{\infty} \frac{1}{\sqrt{2\pi}} \exp\left[ -\frac{1}{2} (x - \rho_{\rm opt}(\theta))^2 \right] dx.
$$

Numerically, the integral for the probability of detection can be evaluated using the cumulative distribution function (CDF) of the standard normal distribution. In practice, this is implemented in `gwsnr` as

```python
P_det = 1 - norm.cdf(snr_th - snr_opt)
```

**Note:** This way of calculating $\rho_{\rm mf}$ doesn't involve matched-filter SNR calculation, but rather uses the optimal SNR $\rho_{\rm opt}$, which is computed using the noise-weighted inner product method or the Partial Scaling method, and then Pdet is derived from the assumption of Gaussian noise.

## Example Usage

Here is an example of how to compute the probability of detection using `gwsnr`:

```python
# loading GWSNR class from the gwsnr package
import gwsnr
import numpy as np

# initializing the GWSNR class with inner product as the signal-to-noise ratio type
gwsnr = gwsnr.GWSNR(snr_type='inner_product', waveform_approximant='IMRPhenomXPHM')

# defining the parameters for the gravitational wave signal for a BBH with GW150914 like parameters
param_dict= dict(
    mass_1=np.array([36.0]), # mass of the primary black hole in solar masses
    mass_2=np.array([29.0]), # mass of the secondary black hole in solar masses
    luminosity_distance=np.array([440.0]), # luminosity distance in Mpc
    theta_jn=np.array([1.0]), # inclination angle in radians
    ra=np.array([3.435]), # right ascension in radians
    dec=np.array([-0.408]), # declination in radians
    psi=np.array([0.0]),  # polarization angle in radians
    geocent_time=np.array([1126259462.4]), # geocentric time in GPS seconds
    a_1=np.array([0.3]), # dimensionless spin of the primary black hole
    a_2=np.array([0.2]), # dimensionless spin of the secondary black hole
    tilt_1=np.array([0.5]), # tilt angle of the primary black hole in radians
    tilt_2=np.array([0.8]), # tilt angle of the secondary black hole in radians
    phi_12=np.array([0.0]), # Relative angle between the primary and secondary spin of the binary in radians
    phi_jl=np.array([0.0]), # Angle between the total angular momentum and the orbital angular momentum in radians
)

# 
# signal-to-noise ratio with detectors LIGO-Hanford, LIGO-Livingston, and Virgo with O4 observing run sensitivity
snr_dict = gwsnr.snr(**param_dict)

# Calculate the probability of detection with a threshold of 8 for matched filter SNR
pdet = gwsnr.probability_of_detection(snr_dict=snr_dict, snr_th=8., snr_th_net=8., type="matched_filter") # or type="bool"
``` -->