import numpy as np
import pytest
from gwsnr import GWSNR
np.random.seed(1234)

class TestGWSNRANN:
    """
    Test suite for validating the hybrid Partial Scaling Interpolation Method with SNR Recalculation
    for accurate Pdet estimation in precessing binary black hole (BBH) systems.
    """

    def test_spinning_bbh_ann(self):
        """
        Test Partial Scaling Interpolation with SNR Recalculation for Precessing BBH Systems.

        This test validates a hybrid approach for SNR and detection probability (Pdet) estimation in
        fully precessing BBH populations. The Partial Scaling method, efficient for aligned-spin systems,
        is augmented here by selectively recalculating SNRs using the Noise-Weighted Inner Product method
        for sources whose interpolated SNRs fall within a predefined range close to the detection threshold.

        Scientific Motivation
        --------------------
        In gravitational-wave astronomy, computational efficiency is crucial for large-scale
        population studies. The Partial Scaling method provides rapid SNR estimates, but may be less
        accurate for generic precessing binaries, especially near the detection threshold ($\rho_{\rm th}$).
        To improve the accuracy of $P_{\rm det}$ without sacrificing speed, SNRs of events with
        interpolated values near threshold are recalculated using the more accurate inner product method.

        Methodology
        -----------
        - Generate a set of generic precessing BBH parameters.
        - Compute network SNR using the Partial Scaling (interpolation_aligned_spins) method.
        - For systems with SNR within the interval [4, 12], recalculate SNR using the Noise-Weighted Inner Product method
          (e.g., via `bilby` or LAL).
        - Validate that, for these systems, the final SNR matches the directly computed inner product SNR.

        Validation Criteria
        ------------------
        - The returned SNR array must be finite, non-negative, and of correct length.
        - For systems in the recalculation range ([4, 12]), the SNR values from the Partial Scaling method
          (after recalculation) must match those from the reference bilby-based inner product computation.
        - Any deviation for these selected events is flagged as a failure.

        This hybrid approach leverages the computational efficiency of interpolation while maintaining high
        fidelity in the most astrophysically relevant regime (near detection threshold), enabling robust and
        efficient $P_{\rm det}$ estimation for population inference and selection bias studies.
        """

        gwsnr = GWSNR(
            npool=4,
            waveform_approximant="IMRPhenomXPHM",
            snr_type="interpolation_aligned_spins",
            snr_recalculation=True,
            snr_recalculation_range=[4,12],
            snr_recalculation_waveform_approximant="IMRPhenomXPHM",
        )

        nsamples = 10
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

        result = gwsnr.snr(gw_param_dict=param_dict)
        # Output validation
        assert isinstance(result, dict), "Output should be a dictionary"
        assert "optimal_snr_net" in result, "'optimal_snr_net' missing in output"

        snr_arr = np.asarray(result["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"SNR shape mismatch: {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values"
        assert np.all(snr_arr >= 0), "Negative SNR values"

        # check with bilby_snr
        bilby_snr = gwsnr.compute_bilby_snr(gw_param_dict=param_dict)["optimal_snr_net"]
        
        # select and check snr between [4, 12] is equal to bilby_snr
        idx = np.where((snr_arr >= 4) & (snr_arr <= 12))[0]
        assert np.all(np.isclose(snr_arr[idx], bilby_snr[idx])), \
            "SNR values do not match between recalculated and bilby_snr for selected range"
