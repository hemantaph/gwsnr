# Probability of Detection Calculation

The detectability of gravitational-wave events is not the same across all sources. Factors such as distance, mass, orbital orientation, and the spins of the compact objects all influence how strong the signal appears in the detector. Nearby or more massive systems, as long as they merge within the detector’s sensitive frequency band, tend to produce stronger signals. The orientation of the binary is also important: signals from systems that are face-on (viewed from above or below their orbital plane) are much stronger than those from edge-on systems. Spins further modify the waveform and can enhance or diminish detectability.

The `gwsnr` package quantifies the likelihood of detecting a signal with a given set of source parameters, $\theta$, through the probability of detection, $P_{\rm det}$. This probability is defined by whether the signal’s signal-to-noise ratio (SNR) is strong enough to stand out from the detector’s background noise. For a given detection threshold, $\rho_{\rm th}$, `gwsnr` implements two models for $P_{\rm det}$.

## Deterministic Threshold (Step Function)

In the simplest model, detection is treated as a binary outcome based on the optimal SNR, $\rho_{\rm opt}$. If the intrinsic SNR exceeds the threshold, the event is considered detected; otherwise, it is not. Mathematically, this step function is expressed as:

$$
P^{\rm step}_{\rm det}(\theta) =
\begin{cases}
1 & \text{if } \rho_{\rm opt}(\theta) > \rho_{\rm th} \\
0 & \text{otherwise}
\end{cases}
$$


## Probabilistic Detection (Gaussian Model)

In practice, a signal is identified as detected if its matched-filter SNR, $\rho'_{\rm mf}$ (where the prime denotes maximization over extrinsic parameters such as sky position and orientation), surpasses a set threshold. A more realistic treatment acknowledges that, due to random noise fluctuations, $\rho'_{\rm mf}$ fluctuates around its true value. Following Thrane & Talbot (2019), the probability distribution of $\rho'_{\rm mf}$ is normally distributed with mean $\rho_{\rm opt}$ and unit variance:

$$
p(\rho'_{\rm mf} \mid \theta) = \frac{1}{\sqrt{2\pi}} \exp\left[ -\frac{1}{2}(\rho'_{\rm mf} - \rho_{\rm opt}(\theta))^2 \right].
$$

The probability of detection in this scenario, $P^{\rm Gauss}_{\rm det}$, is the probability that the measured SNR exceeds the threshold $\rho_{\rm th}$. This is calculated by integrating the probability density from the threshold to infinity:

$$
\begin{align}
P^{\rm Gauss}_{\rm det}(\theta) &= \int_{\rho_{\rm th}}^{\infty} p(\rho'_{\rm mf} \mid \theta) d\rho'_{\rm mf}\,, \notag \\
&= \int_{\rho_{\rm th}}^{\infty} \frac{1}{\sqrt{2\pi}} \exp\left[ -\frac{1}{2}(x - \rho_{\rm opt}(\theta))^2 \right] dx\,, \notag \\
&= \frac{1}{2} {\rm erfc}\left( \frac{\rho_{\rm th} - \rho_{\rm opt}(\theta)}{\sqrt{2}} \right)\,.
\end{align}
$$

In `gwsnr`, this probability is efficiently computed using the cumulative distribution function (CDF) of the standard normal distribution.


## Implementation

It is important to clarify that `gwsnr` does not perform a separate matched-filter analysis to find $\rho'_{\rm mf}$, as it is not required to get $P_{\rm det}$ evident from Eqn.(3). Instead, it first calculates the optimal SNR ($\rho_{\rm opt}$) using one of its highly efficient methods (such as Partial Scaling or the noise-weighted inner product). This $\rho_{\rm opt}$ value is then used as the mean in the Gaussian model above to derive a realistic detection probability. The implementation in `gwsnr` is equivalent to:

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

print("SNRs computed with inner product:\n", snr_dict)
print("Probability of detection (deterministic):\n", pdet_bool)
print("Probability of detection (probabilistic):\n", pdet_gaussian)
``` 

**Expected Output:**

```
SNRs computed with inner product:
 {'L1': array([ 8.20570674,  6.6467511 , 11.8892919 ,  0.        ]), 'H1': array([6.87033936, 5.48562041, 9.66843147, 0.        ]), 'V1': array([3.56751442, 2.64453924, 4.71463604, 0.        ]), 'optimal_snr_net': array([11.28106135,  9.01470572, 16.03314136,  0.        ])}
Probability of detection (deterministic):
 {'L1': array([1, 0, 1, 0]), 'H1': array([0, 0, 1, 0]), 'V1': array([0, 0, 0, 0]), 'pdet_net': array([1, 1, 1, 0])}
Probability of detection (probabilistic):
 {'L1': array([5.81490005e-01, 8.79880633e-02, 9.99949731e-01,
       6.66133815e-16]), 'H1': array([1.29309625e-01, 5.96210075e-03, 9.52384947e-01,
       6.66133815e-16]), 'V1': array([4.65764666e-06, 4.26693338e-08, 5.09253562e-04,
       6.66133815e-16]), 'pdet_net': array([9.99482914e-01, 8.44876937e-01, 1.00000000e+00,
       6.66133815e-16])}
```

## Visualization of Detection Probability

<div align="center">
<figure>
    <img src="_static/pdet_comparison.png" alt="Probability of Detection Comparison" width="600"/>
    <figcaption align="left"><b>Figure:</b> Comparison of two models for the gravitational-wave probability of detection, $P_{\rm det}$, as a function of the optimal signal-to-noise ratio, $\rho_{\rm opt}$. The deterministic model (blue line) is a step function: detection is certain ($P_{\rm det} = 1$) only if $\rho_{\rm opt}$ exceeds the threshold ($\rho_{\rm th} = 8$, red dashed line). In contrast, the probabilistic model (orange curve) accounts for Gaussian noise fluctuations in the measured SNR, resulting in a smooth transition in detection probability. For an example signal with $\rho_{\rm opt} = 8.2$ (green dashed line), the deterministic model yields $P_{\rm det} = 1$, while the more realistic probabilistic model gives a detection probability of $P_{\rm det} \approx 0.58$.
    </figcaption>
</figure>
</div>


<!-- ## Probability of Detection Calculation

The `gwsnr` package provides tools to evaluate the probability of detecting a GW signal, denoted as $P_{\rm det}$. The calculation is based on whether the observed SNR exceeds a specified threshold, $\rho_{\rm th}$, for either individual detectors or a detector network. For most practical applications with Gaussian noise, using an SNR threshold is a reliable (proxy) criterion for detection.

Within gwsnr, two principal approaches are available for computing $P_{\rm det}$: one based on the optimal SNR, $\rho_{\rm opt}$, and another that considers the statistical nature of the matched-filter SNR, $\rho'_{\rm mf}$.

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

For a more realistic scenario, the matched-filter SNR, $\rho'_{\rm mf}$, fluctuates due to noise and follows a normal distribution with mean $\rho_{\rm opt}(\theta)$ and unit variance for a given set of parameters $\theta$. Following [Thrane et al. 2019](https://arxiv.org/abs/1809.02293), the probability density for measuring a particular value $\rho'_{\rm mf}$ is

$$
p(\rho'_{\rm mf} | \theta) = \frac{1}{\sqrt{2\pi}} \exp\left[-\frac{1}{2} \left( \rho'_{\rm mf} - \rho_{\rm opt}(\theta) \right)^2 \right].
$$

The probability that the measured SNR exceeds the threshold, i.e., the probability of detection, is then given by

$$
P_{\rm det} = P(\theta\mid \rho_{\rm opt}) = \int_{\rho_{\rm th}}^{\infty} \frac{1}{\sqrt{2\pi}} \exp\left[ -\frac{1}{2} (x - \rho_{\rm opt}(\theta))^2 \right] dx.
$$

Numerically, the integral for the probability of detection can be evaluated using the cumulative distribution function (CDF) of the standard normal distribution. In practice, this is implemented in `gwsnr` as

```python
P_det = 1 - norm.cdf(snr_th - snr_opt)
```

**Note:** This way of calculating $\rho'_{\rm mf}$ doesn't involve matched-filter SNR calculation, but rather uses the optimal SNR $\rho_{\rm opt}$, which is computed using the noise-weighted inner product method or the Partial Scaling method, and then Pdet is derived from the assumption of Gaussian noise.

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