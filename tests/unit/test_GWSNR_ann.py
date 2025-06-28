import numpy as np
import pytest
from gwsnr import GWSNR

np.random.seed(1234)

class TestGWSNRANN:
    """
    Test suite for validating the GWSNR Artificial Neural Network (ANN) model
    for signal-to-noise ratio (SNR) and detection flag (`pdet_net`) calculation
    in binary black hole (BBH) systems. These tests assess the consistency,
    physical validity, and reproducibility of the ANN model, and benchmark its
    detection outputs against the direct inner product method.
    """

    def test_spinning_bbh_ann(self):
        """
        Validate SNR generation using the ANN method for spinning, fully precessing BBH systems.

        This test generates a set of physically realistic precessing BBH parameters and computes
        the network SNR using the ANN model (`snr_type="ann"`). The output is validated for physical
        correctness (finite, non-negative values, correct array shapes) and tested for numerical
        reproducibility (identical results for identical input). This test serves as a basic
        correctness check for the ANN inference pipeline in the GWSNR framework.

        Scientific Purpose
        ------------------
        Ensures that the ANN model for SNR prediction is robust and reliable when used for
        generic precessing BBH populations. Numerical reproducibility is essential for
        Bayesian inference and population synthesis studies.

        Validation Criteria
        -------------------
        - Output must be a dictionary containing the key `'optimal_snr_net'`.
        - SNR array must match the number of input samples, be finite, and non-negative.
        - Repeated evaluation with identical input must yield numerically identical results.
        """
        gwsnr = GWSNR(
            npool=4,
            waveform_approximant="IMRPhenomXPHM",
            snr_type="ann",
            pdet=False,
        )

        nsamples = 5
        mtot = np.random.uniform(2*4.98, 2*112.5, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            mass_1=mtot / (1 + mass_ratio),
            mass_2=mtot * mass_ratio / (1 + mass_ratio),
            luminosity_distance=500 * np.ones(nsamples),
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
            a_1=np.random.uniform(0, 0.8, size=nsamples),
            a_2=np.random.uniform(0, 0.8, size=nsamples),
            tilt_1=np.random.uniform(0, np.pi, size=nsamples),
            tilt_2=np.random.uniform(0, np.pi, size=nsamples),
            phi_12=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phi_jl=np.random.uniform(0, 2 * np.pi, size=nsamples),
        )

        ann_result = gwsnr.snr(gw_param_dict=param_dict)
        # Output validation
        assert isinstance(ann_result, dict), "Output should be a dictionary"
        assert "optimal_snr_net" in ann_result, "'optimal_snr_net' missing in ANN output"

        snr_arr = np.asarray(ann_result["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"SNR shape mismatch: {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values"
        assert np.all(snr_arr >= 0), "Negative SNR values"

        # Reproducibility check
        ann_result2 = gwsnr.snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            snr_arr, np.asarray(ann_result2["optimal_snr_net"]), rtol=1e-10
        )

    def test_ann_vs_inner_product_pdet_exact(self):
        """
        Test for strict agreement of detection flags (`pdet_net`) between ANN and inner product methods.

        This test generates a set of spinning BBH parameters, evaluates the detection
        flag (`pdet_net`, with values 0 or 1) using both the ANN-based and the direct
        inner product method, and asserts exact element-wise equality for all samples.

        Scientific Purpose
        ------------------
        This test provides a rigorous consistency check for the binary classification
        (`pdet_net`) implemented in the ANN model, benchmarked against the reference
        physical inner product approach. Such consistency is required for high-confidence
        population inference, event rate estimation, and statistical detection studies.

        Validation Criteria
        -------------------
        - The output dictionary from the ANN method must contain the key `'pdet_net'`.
        - The output array must be binary (strictly 0 or 1) and match the number of samples.
        - All values of `pdet_net` from the ANN and inner product methods must agree exactly.
        - If any disagreement is found, the test fails and reports the mismatched indices.

        Note
        ----
        This test enforces exact equivalence of event-wise detection decisions, as required for
        reproducible and unbiased detection statistics in gravitational-wave astrophysics.
        """
        nsamples = 20
        mtot = np.random.uniform(2*4.98, 2*112.5, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            mass_1=mtot / (1 + mass_ratio),
            mass_2=mtot * mass_ratio / (1 + mass_ratio),
            luminosity_distance=500 * np.ones(nsamples),
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(-np.pi/2, np.pi/2, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
            a_1=np.random.uniform(0, 0.8, size=nsamples),
            a_2=np.random.uniform(0, 0.8, size=nsamples),
            tilt_1=np.random.uniform(0, np.pi, size=nsamples),
            tilt_2=np.random.uniform(0, np.pi, size=nsamples),
            phi_12=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phi_jl=np.random.uniform(0, 2 * np.pi, size=nsamples),
        )

        gwsnr_ann = GWSNR(
            npool=2,
            waveform_approximant="IMRPhenomXPHM",
            snr_type="ann",
            pdet=True,
        )
        ann_out = gwsnr_ann.snr(gw_param_dict=param_dict)

        # Output validation
        assert isinstance(ann_out, dict), "Output should be a dictionary"
        assert "pdet_net" in ann_out, "'pdet_net' missing in ANN output"
        pdet_arr = np.asarray(ann_out["pdet_net"])
        assert pdet_arr.shape == (nsamples,), f"pdet shape mismatch: {pdet_arr.shape}"
        assert np.all(np.logical_or(pdet_arr == 0, pdet_arr == 1)), "pdet values must be 0 or 1"

        # Compare with inner product method
        gwsnr_ip = GWSNR(
            npool=2,
            waveform_approximant="IMRPhenomXPHM",
            snr_type="inner_product",
            pdet=True,
        )

        ip_out = gwsnr_ip.snr(gw_param_dict=param_dict)

        pdet_ann = np.asarray(ann_out["pdet_net"])
        pdet_ip = np.asarray(ip_out["pdet_net"])
        assert pdet_ann.shape == pdet_ip.shape
        np.testing.assert_array_equal(
            pdet_ann, pdet_ip,
            err_msg="Mismatch: ANN and inner product pdet_net do not agree exactly."
        )
