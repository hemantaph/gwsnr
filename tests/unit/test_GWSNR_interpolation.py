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
        Considers spinless BBH systems 
        """
        gwsnr = GWSNR(
            npool=int(4),
            mtot_min=2*4.98, # 4.98 Mo is the minimum component mass of BBH systems in GWTC-3
            mtot_max=2*112.5+10.0, # 112.5 Mo is the maximum component mass of BBH systems in GWTC-3. 10.0 Mo is added to avoid edge effects.
            ratio_min=0.1,
            ratio_max=1.0,
            spin_max=0.99,
            mtot_resolution=200,
            ratio_resolution=20,
            spin_resolution=10,
            sampling_frequency=2048.0,
            waveform_approximant="IMRPhenomD",
            frequency_domain_source_model='lal_binary_black_hole',
            minimum_frequency=20.0,
            duration_max=None,
            duration_min=None,
            snr_type="interpolation_no_spins",
            psds=None,
            ifos=None,
            interpolator_dir="./interpolator_pickle",
            create_new_interpolator=False,
            gwsnr_verbose=True,
            multiprocessing_verbose=True,
            mtot_cut=False,
            pdet=False,
            snr_th=8.0,
            snr_th_net=8.0,
            ann_path_dict=None,
            snr_recalculation=False,
            snr_recalculation_range=[6,8],
            snr_recalculation_waveform_approximant="IMRPhenomXPHM",
        )

        nsamples = 5
        mtot = np.random.uniform(2*4.98, 2*112.5, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            # convert to component masses
            mass_1 = mtot / (1 + mass_ratio),
            mass_2 = mtot * mass_ratio / (1 + mass_ratio),
            # Fix luminosity distance
            luminosity_distance = 500*np.ones(nsamples),
            # Randomly sample everything else:
            theta_jn = np.random.uniform(0, 2*np.pi, size=nsamples),
            ra = np.random.uniform(0, 2*np.pi, size=nsamples), 
            dec = np.random.uniform(-np.pi/2, np.pi/2, size=nsamples), 
            psi = np.random.uniform(0, 2*np.pi, size=nsamples),
            phase = np.random.uniform(0, 2*np.pi, size=nsamples),
            geocent_time = 1246527224.169434*np.ones(nsamples),
        )
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Output checks
        assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"
        
        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values present"
        assert np.all(np.isreal(snr_arr)), "SNR values should be real numbers"
        assert np.all(snr_arr >= 0), "SNR should not be negative"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"

        # JSON output
        param_dict.update(interp_snr)
        output_file = tmp_path / "snr_data_interpolation.json"
        append_json(output_file, param_dict, replace=True)
        assert output_file.exists(), "Output JSON file was not created"
        assert output_file.stat().st_size > 0, "Output file is empty"

        # Reproducibility check
        interp_snr2 = gwsnr.snr(gw_param_dict=param_dict)
        np.testing.assert_allclose(
            interp_snr["optimal_snr_net"],
            interp_snr2["optimal_snr_net"],
            rtol=1e-10,
            err_msg="SNR calculation is not deterministic"
        )

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        gwsnr = GWSNR(snr_type="interpolation_no_spins", mtot_resolution=20, ratio_resolution=5, gwsnr_verbose=False)
        
        # Test with mismatched array lengths
        # Mismatched lengths is allowed in the current implementation
        # with pytest.raises((ValueError, IndexError, AssertionError)):
        #     gwsnr.snr(gw_param_dict=dict(
        #         mass_1=np.array([30, 40]),
        #         mass_2=np.array([20]),  # Different length
        #         luminosity_distance=np.array([400, 500]),
        #     ))
        
        # Test with negative masses
        with pytest.raises((ValueError, AssertionError)):
            gwsnr.snr(gw_param_dict=dict(
                mass_1=np.array([-30]),
                mass_2=np.array([20]),
                luminosity_distance=np.array([400]),
            ))

        # Test with np.nan values
        with pytest.raises((TypeError, ValueError)):
            gwsnr.snr(gw_param_dict=dict(
                mass_1=np.array([30, 40]),
                mass_2=np.array([20, np.nan]),
                luminosity_distance=np.array([400, 500]),
            ))

        # Test with np.inf values
        with pytest.raises((TypeError, ValueError)):
            gwsnr.snr(gw_param_dict=dict(
                mass_1=np.array([30, 40]),
                mass_2=np.array([20, np.inf]),
                luminosity_distance=np.array([400, 500]),
            ))

        # Test with empty arrays
        with pytest.raises((ValueError, AssertionError)):
            gwsnr.snr(gw_param_dict=dict(
                mass_1=np.array([]),
                mass_2=np.array([]),
                luminosity_distance=np.array([]),
            ))

    def test_single_event(self):
        """Test SNR calculation for a single event."""
        gwsnr = GWSNR(snr_type="interpolation_no_spins", mtot_resolution=20, ratio_resolution=5, gwsnr_verbose=False)
        
        param_dict = dict(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=400.0,
            theta_jn=0.5,
            ra=1.0,
            dec=0.2,
            psi=0.3,
            phase=0.1,
            geocent_time=1246527224.169434,
        )
        
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)
        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        
        assert snr_arr.shape == (1,), "Single event should return single SNR value"
        assert np.isfinite(snr_arr[0]), "Single event SNR should be finite"
        assert snr_arr[0] >= 0, "Single event SNR should be non-negative"

    def test_snr_generation_for_various_interpolation(self):
        """
        Test SNR generation with various GWSNR interpolation methods.
        Interpolation methods include:
        - interpolation_no_spins  (already tested above in test_default_snr_generation)
        - interpolation_no_spins_jax
        - interpolation_aligned_spins
        - interpolation_aligned_spins_jax
        """

        nsamples = 5
        mtot = np.random.uniform(2*4.98, 2*112.5, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            # convert to component masses
            mass_1 = mtot / (1 + mass_ratio),
            mass_2 = mtot * mass_ratio / (1 + mass_ratio),
            # Fix luminosity distance
            luminosity_distance = 500*np.ones(nsamples),
            # Randomly sample everything else:
            theta_jn = np.random.uniform(0, 2*np.pi, size=nsamples),
            ra = np.random.uniform(0, 2*np.pi, size=nsamples), 
            dec = np.random.uniform(-np.pi/2, np.pi/2, size=nsamples), 
            psi = np.random.uniform(0, 2*np.pi, size=nsamples),
            phase = np.random.uniform(0, 2*np.pi, size=nsamples),
            geocent_time = 1246527224.169434*np.ones(nsamples),
        )

        # Test with interpolation_no_spins_jax
        gwsnr = GWSNR(snr_type="interpolation_no_spins_jax", mtot_resolution=20, ratio_resolution=5, gwsnr_verbose=False)
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Output checks
        assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"
        
        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values present"
        assert np.all(np.isreal(snr_arr)), "SNR values should be real numbers"
        assert np.all(snr_arr >= 0), "SNR should not be negative"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"

        # add spins to the parameters
        param_dict["a_1"] = np.random.uniform(-0.8,0.8, size=nsamples)
        param_dict["a_2"] = np.random.uniform(-0.8,0.8, size=nsamples)

        # Test with interpolation_aligned_spins
        gwsnr = GWSNR(snr_type="interpolation_aligned_spins", mtot_resolution=20, ratio_resolution=5, spin_resolution=5, gwsnr_verbose=False)
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)
        
        # Output checks
        assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"
        
        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values present"
        assert np.all(np.isreal(snr_arr)), "SNR values should be real numbers"
        assert np.all(snr_arr >= 0), "SNR should not be negative"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"

        # Test with interpolation_aligned_spins_jax
        gwsnr = GWSNR(snr_type="interpolation_aligned_spins_jax", mtot_resolution=20, ratio_resolution=5, spin_resolution=5, gwsnr_verbose=False)
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Output checks
        assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"
        
        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values present"
        assert np.all(np.isreal(snr_arr)), "SNR values should be real numbers"
        assert np.all(snr_arr >= 0), "SNR should not be negative"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"

    def test_snr_generation_custom_input_arguments(self):
        """
        Test SNR generation with various GWSNR interpolation methods.
        Interpolation methods include:
        - interpolation_no_spins  (already tested above in test_default_snr_generation)
        - interpolation_no_spins_jax
        - interpolation_aligned_spins
        - interpolation_aligned_spins_jax
        """

        nsamples = 5
        mtot = np.random.uniform(2*1.0, 2*3.0, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            # convert to component masses
            mass_1 = mtot / (1 + mass_ratio),
            mass_2 = mtot * mass_ratio / (1 + mass_ratio),
            # Fix luminosity distance
            luminosity_distance = 100*np.ones(nsamples),
            # Randomly sample everything else:
            theta_jn = np.random.uniform(0, 2*np.pi, size=nsamples),
            ra = np.random.uniform(0, 2*np.pi, size=nsamples), 
            dec = np.random.uniform(-np.pi/2, np.pi/2, size=nsamples), 
            psi = np.random.uniform(0, 2*np.pi, size=nsamples),
            phase = np.random.uniform(0, 2*np.pi, size=nsamples),
            geocent_time = 1246527224.169434*np.ones(nsamples),
        )

        # Test with interpolation_no_spins_jax
        gwsnr = GWSNR(
            npool=int(4),
            mtot_min=2*1.0, # 4.98 Mo is the minimum component mass of BBH systems in GWTC-3
            mtot_max=2*3.0+1.0, # 112.5 Mo is the maximum component mass of BBH systems in GWTC-3. 10.0 Mo is added to avoid edge effects.
            ratio_min=0.1,
            ratio_max=1.0,
            spin_max=0.99,
            mtot_resolution=20,
            ratio_resolution=5,
            spin_resolution=5,
            sampling_frequency=1024.0,
            waveform_approximant="TaylorF2",
            frequency_domain_source_model='lal_binary_black_hole',
            minimum_frequency=30.0,
            snr_type="interpolation_no_spins",
            ifos=['CE'],
            mtot_cut=True,
        )

        interp_snr = gwsnr.snr(gw_param_dict=param_dict)
        # Output checks
        assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
        assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"

        snr_arr = np.asarray(interp_snr["optimal_snr_net"])
        assert snr_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {snr_arr.shape}"
        assert np.all(np.isfinite(snr_arr)), "Non-finite SNR values present"
        assert np.all(np.isreal(snr_arr)), "SNR values should be real numbers"
        assert np.all(snr_arr >= 0), "SNR should not be negative"
        assert snr_arr.dtype == np.float64, f"Expected float64, got {snr_arr.dtype}"

    def test_pdet_generation(self):
        """
        Test probability of detection (pdet) generation with GWSNR interpolation methods.
        """

        nsamples = 5
        mtot = np.random.uniform(2*4.98, 2*112.5, nsamples)
        mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
        param_dict = dict(
            # convert to component masses
            mass_1 = mtot / (1 + mass_ratio),
            mass_2 = mtot * mass_ratio / (1 + mass_ratio),
            # Fix luminosity distance
            luminosity_distance = 500*np.ones(nsamples),
            # Randomly sample everything else:
            theta_jn = np.random.uniform(0, 2*np.pi, size=nsamples),
            ra = np.random.uniform(0, 2*np.pi, size=nsamples), 
            dec = np.random.uniform(-np.pi/2, np.pi/2, size=nsamples), 
            psi = np.random.uniform(0, 2*np.pi, size=nsamples),
            phase = np.random.uniform(0, 2*np.pi, size=nsamples),
            geocent_time = 1246527224.169434*np.ones(nsamples),
        )

        # Test with interpolation_no_spins_jax, pdet=True
        gwsnr = GWSNR(snr_type="interpolation_no_spins", mtot_resolution=20, ratio_resolution=5, gwsnr_verbose=False, pdet=True, snr_th=8.0, snr_th_net=8.0)
        interp_pdet = gwsnr.snr(gw_param_dict=param_dict)

        # Output checks
        assert isinstance(interp_pdet, dict), "pdet output should be a dictionary"
        assert "pdet_net" in interp_pdet, "Expected 'pdet_net' in pdet output"
        # check pdet is 0 or 1
        pdet_arr = np.asarray(interp_pdet["pdet_net"])
        assert pdet_arr.shape == (nsamples,), f"Expected shape ({nsamples},), got {pdet_arr.shape}"
        assert np.all(np.isin(pdet_arr, [0, 1])), "pdet values should be binary (0 or 1)"

        # find SNR values first and then check pdet
        gwsnr = GWSNR(snr_type="interpolation_no_spins", gwsnr_verbose=False, pdet=False)
        interp_snr = gwsnr.snr(gw_param_dict=param_dict)

        # Pdet wrt optimal SNR
        interp_pdet_optimal = gwsnr.probability_of_detection(snr_dict=interp_snr, snr_th=8.0, snr_th_net=8.0, type='bool')
        # Pdet wrt matched-filter SNR with the assumption of gaussian noise
        interp_pdet_match_filter = gwsnr.probability_of_detection(snr_dict=interp_snr, snr_th=8.0, snr_th_net=8.0, type='matched_filter')

        # Output checks
        # check interp_pdet_optimal is 0 or 1
        assert np.all(np.isin(interp_pdet_optimal["pdet_net"], [0, 1])), "pdet values should be binary (0 or 1)"
        # check interp_pdet_match_filter is within 0 and 1
        assert np.all((interp_pdet_match_filter["pdet_net"] >= 0) & (interp_pdet_match_filter["pdet_net"] <= 1)), "pdet values should be within [0, 1]"