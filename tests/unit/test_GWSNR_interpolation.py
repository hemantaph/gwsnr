import numpy as np
import pytest
from gwsnr import GWSNR
from gwsnr.utils import append_json

np.random.seed(1234)

class TestGWSNRInterpolation:
    """
    Unit tests for the GWSNR interpolation-based SNR calculation.
    """

    def test_default_snr_generation(self, tmp_path):
        """
        Test SNR generation with default GWSNR interpolation settings.
        """
        gwsnr = GWSNR(
            npool=int(4),
            mtot_min=2.0,
            mtot_max=439.6,
            ratio_min=0.1,
            ratio_max=1.0,
            spin_max=0.9,
            mtot_resolution=200,
            ratio_resolution=50,
            spin_resolution=20,
            sampling_frequency=2048.0,
            waveform_approximant="IMRPhenomD",
            frequency_domain_source_model='lal_binary_black_hole',
            minimum_frequency=20.0,
            duration_max=None,
            duration_min=None,
            snr_type="interpolation",
            psds=None,
            ifos=None,
            interpolator_dir="./interpolator_pickle",
            create_new_interpolator=False,
            gwsnr_verbose=True,
            multiprocessing_verbose=True,
            mtot_cut=True,
            pdet=False,
            snr_th=8.0,
            snr_th_net=8.0,
            ann_path_dict=None,
            snr_recalculation=False,
            snr_recalculation_range=[6,8],
            snr_recalculation_waveform_approximant="IMRPhenomXPHM",
        )

        nsamples = 10
        chirp_mass = np.linspace(5, 60, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            mass_1=(chirp_mass * (1 + mass_ratio) ** (1 / 5)) / mass_ratio ** (3 / 5),
            mass_2=chirp_mass * mass_ratio ** (2 / 5) * (1 + mass_ratio) ** (1 / 5),
            luminosity_distance=500 * np.ones(nsamples),
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(0, np.pi, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
        )

        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Output checks
        assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"
        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), "Unexpected SNR array shape"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values present"
        assert np.all(snr_arr >= 0), "SNR should not be negative"

        m1 = param_dict["mass_1"]
        m2 = param_dict["mass_2"]
        assert np.all(m1 >= m2), "mass_1 should be >= mass_2"

        # JSON output
        param_dict.update(interp_snr)
        output_file = tmp_path / "snr_data_interpolation.json"
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists(), "Output JSON file was not created"

        # Optional: reproducibility check (if method is deterministic)
        interp_snr2 = gwsnr.snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            interp_snr["optimal_snr_net"],
            interp_snr2["optimal_snr_net"],
            rtol=1e-10,
        )

    def test_different_waveform_approximant(self, tmp_path):
        """
        Test SNR interpolation with a different waveform approximant.
        """
        gwsnr = GWSNR(
            npool=2,
            mtot_resolution=50,
            ratio_resolution=10,
            waveform_approximant="SEOBNRv4_ROM",
            snr_type="interpolation",
            ifos=["L1"],
            interpolator_dir="./interpolator_pickle",
            create_new_interpolator=False,
            gwsnr_verbose=False,
            multiprocessing_verbose=False,
        )

        nsamples = 5
        chirp_mass = np.linspace(10, 40, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            mass_1=(chirp_mass * (1 + mass_ratio) ** (1 / 5)) / mass_ratio ** (3 / 5),
            mass_2=chirp_mass * mass_ratio ** (2 / 5) * (1 + mass_ratio) ** (1 / 5),
            luminosity_distance=800 * np.ones(nsamples),
            theta_jn=np.random.uniform(0, 2 * np.pi, size=nsamples),
            ra=np.random.uniform(0, 2 * np.pi, size=nsamples),
            dec=np.random.uniform(0, np.pi, size=nsamples),
            psi=np.random.uniform(0, 2 * np.pi, size=nsamples),
            phase=np.random.uniform(0, 2 * np.pi, size=nsamples),
            geocent_time=1246527224.169434 * np.ones(nsamples),
        )

        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        assert isinstance(interp_snr, dict)
        assert "optimal_snr_net" in interp_snr
        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,)
        assert np.all(np.isfinite(snr_arr))
        assert np.all(snr_arr >= 0)

        # JSON output
        param_dict.update(interp_snr)
        output_file = tmp_path / "snr_data_interpolation_seobnrv4.json"
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists()

    # def test_zero_samples(self):
    #     """
    #     Test GWSNR interpolation method with zero samples (edge case).
    #     """
    #     gwsnr = GWSNR(
    #         npool=1,
    #         mtot_resolution=10,
    #         ratio_resolution=5,
    #         waveform_approximant="IMRPhenomD",
    #         snr_type="interpolation",
    #         ifos=["L1"],
    #         interpolator_dir="./interpolator_pickle",
    #         create_new_interpolator=False,
    #         gwsnr_verbose=False,
    #         multiprocessing_verbose=False,
    #     )

    #     param_dict = dict(
    #         mass_1=np.array([]),
    #         mass_2=np.array([]),
    #         luminosity_distance=np.array([]),
    #         theta_jn=np.array([]),
    #         ra=np.array([]),
    #         dec=np.array([]),
    #         psi=np.array([]),
    #         phase=np.array([]),
    #         geocent_time=np.array([]),
    #     )

    #     interp_snr = gwsnr.snr(gw_param_dict=param_dict)
    #     assert isinstance(interp_snr, dict)
    #     assert "optimal_snr_net" in interp_snr
    #     assert len(interp_snr["optimal_snr_net"]) == 0

    # You can add more tests, e.g., invalid input types, missing keys, extreme mass ratios, etc.

