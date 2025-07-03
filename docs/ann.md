# ANN-based Pdet Estimation

The `gwsnr` package now incorporates an artificial neural network (ANN) model, developed using TensorFlow (@tensorflow:2015) and scikit-learn (@scikitlearn:2011), to rapidly estimate $P_{\rm det}$ in binary black hole (BBH) systems using the IMRPhenomXPHM waveform approximant. This complex IMR waveform model accounts for spin-precessing systems with subdominant harmonics. The ANN model is especially useful when precise signal-to-noise ratio (SNR) calculations are not critical, providing a quick and effective means of estimating $P_{\rm det}$. This value indicates detectability under Gaussian noise by determining if the SNR exceeds a certain threshold (e.g., $\rho_{\rm th}=8$). Trained on a large dataset from the `ler` package, the ANN model uses 'partial scaled SNR' values as a primary input, reducing input dimensionality from 15 to 5 and enhancing accuracy. This approach offers a practical solution for assessing detectability under specified conditions. Other similar efforts with ANN models are detailed in (@ChapmanBird:2023, @Gerosa:2020, @Callister:2024).

## Data generation

using the `ler` package, a large dataset of compact binary systems is generated, covering the 15 parameters relevant for the IMRPhenomXPHM waveform model. This includes Gravitational wave source properties: $m_1$ (mass of the primary black hole), $m_2$ (mass of the secondary black hole), $d_L$ (luminosity distance), $\iota$ (inclination-angle), ($a_1, a_2$) (dimensionless spin of the primary and secondary black hole), $a_2$ (dimensionless spin of the secondary black hole), ($\theta_1, \theta_2$) (tilt angle of the primary and secondary black hole), $\delta \phi$ (relative angle between the primary and secondary spin of the binary), $\phi_{\rm JL}$ (angle between total and orbital angular momentum), ($ra, dec$) (right ascension and declination), $\psi$ (polarization angle), $\phi_c$ (coalescence phase), and $t_c$ (geocentric time). With appropriate waveform approximant and noise model, SNR values are computed with inner product method, which is the most accurate method for computing SNRs in `gwsnr`. The dataset is generated with a range of SNR values, particularly around the threshold SNR of 8, to ensure the ANN model can generate accurate estimates of SNR and then $P_{\rm det}$ can be computed from the SNR values accurately. The dataset is then stored in a json file. At least 100,000 samples are generated to ensure a robust training dataset. The dataset is then split into training and testing sets, with 90% of the data used for training and 10% for testing. The training set is used to train the ANN model, while the testing set is used to evaluate the model's performance.

Increasing the number of samples in the dataset improves the accuracy of the ANN model.

## ANN Model Architecture and Training

- Creating input data: The generated dataset is used to create input data for the ANN model. The input data consists of: Partial-SNR $\rho_{\frac{1}{2}}$, Amplitue*  







## Training Custom ANN Models

In addition to providing trained ANN models for specific configurations, `gwsnr` offers users the flexibility to develop and train custom models tailored to their unique requirements. This adaptability allows for optimization based on variations in detector sensitivity, gravitational-wave properties, and other research-specific factors, ensuring maximum model effectiveness across different scenarios.

















<div align="center">
<figure>
    <img src="_static/ann.png" alt="ANN model architecture" />
    <figcaption>
        <b>Figure.</b> ANN model architecture for estimating $P_{\rm det}$ in binary black hole (BBH) systems. The model takes 'partial scaled SNR' values as input and outputs the estimated detection probability. This approach significantly reduces input dimensionality and enhances estimation accuracy.
    </figcaption>
</figure>
</div>