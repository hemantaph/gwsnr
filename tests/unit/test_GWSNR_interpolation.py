import numpy as np
import pytest
from gwsnr import GWSNR

np.random.seed(1234)

class TestGWSNRANN:
    """
    Test suite for the GWSNR Artificial Neural Network (ANN) model for rapid $P_{\rm det}$ (detectability)
    estimation in binary black hole (BBH) systems using the IMRPhenomXPHM waveform. The ANN model leverages
    partial scaled SNR and reduced dimensionality inputs for efficient classification of detectable events
    under Gaussian detector noise.

    Background
    ----------
    The ANN implementation in `gwsnr` is built with TensorFlow (@tensorflow:2015) and scikit-learn (@scikitlearn:2011)
    and is trained on large datasets from the `ler` package. It is designed to offer fast and practical $P_{\rm det}$
    estimates, especially in scenarios where precise SNR calculations are not essential but a reliable detectability
    assessment is required. The model primarily uses a compressed, 5-dimensional representation of 15 physical
    parameters, capturing the most predictive features for detection in realistic search pipelines.
    """

    def test_spinning_bbh_ann(self):
        """
        Validate ANN-based SNR and $P_{\rm det}$ estimation for generic precessing BBH systems.

        This test generates astrophysically realistic BBH parameters and applies the `gwsnr` ANN model
        (`snr_type="ann"`) to compute network SNR and the binary detection flag (`pdet_net`). The test
        ensures output validity (finite, non-negative SNR), shape correctness, and full reproducibility
        under repeated evaluation.

        Scientific Context
        -----------------
        The ANN model is trained to rapidly approximate the detectability of BBH signals, using the
        IMRPhenomXPHM waveform. It is particularly effective when thousands to millions of parameter
        samples must be classified for detectability in large-scale population or selection-bias studies.
        Precise SNR is not required in such regimes; instead, reliable binary classification (detected
        vs not detected) is prioritized.

        Key Features Tested
        -------------------
        - Model inference on precessing systems, with 15-parameter input vector reduced to 5 via partial scaling.
        - Return of 'optimal_snr_net' for each event.
        - Reproducibility of ANN inference (determinism).
        - All SNR values are physically valid.

        References
        ----------
        @tensorflow:2015, @scikitlearn:2011, @ChapmanBird:2023, @Gerosa:2020, @Callister:2024
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
        Test strict agreement of the binary detection flag (`pdet_net`) between the ANN model and
        the reference noise-weighted inner product method.

        This test generates precessing BBH parameter samples and evaluates detectability via both
        the ANN and the traditional inner product method. It asserts that for every sample, the
        binary detection flag (0 or 1) is *identical* between the ANN and the reference calculation.

        Scientific Rationale
        -------------------
        As the ANN is trained to approximate the detectability threshold ($\rho_{\rm th}$, typically 8)
        as defined by the more computationally expensive inner product, perfect agreement is expected
        and required. This test ensures that the ANN can be safely used for high-throughput selection
        bias and rate calculations where fast, binary decisions are needed.

        Validation
        ----------
        - Output from both methods must contain `pdet_net`.
        - All outputs must be strictly binary (0 or 1).
        - Elementwise agreement is required for all events.

        References
        ----------
        See main ANN documentation and @ChapmanBird:2023, @Gerosa:2020, @Callister:2024 for
        related efforts on neural-network-based GW detectability estimation.
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
