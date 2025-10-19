"""
Unit Tests for SNRThresholdFinder (Threshold optimization via cross-entropy)

This suite verifies:
- Core API behavior with in-memory arrays (no catalog file)
- HDF5 catalog ingestion and preprocessing (selection ranges, key mapping)
- Threshold optimization routine find_best_SNR_threshold on a known shape
- Returned shapes, types, bounds, and basic numerical sanity
- Error handling when required parameters are missing for array mode

Notes:
- We disable multiprocessing in tests for determinism and to avoid runner quirks.
- Synthetic 1D data is used to keep tests fast and hermetic (no external files).
"""

import os
import sys
import tempfile
import numpy as np
import pytest


class TestSNRThresholdFinder:


        # Configure the finder with arrays directly (catalog_file=None)
        finder = SNRThresholdFinder(
            catalog_file=None,
            npool=2,
            multiprocessing_verbose=False,
            selection_range=dict(
                key_name="mass1_source",
                parameter=data["mass1_source"],
                range=(5.0, 200.0),
            ),
            original_detection_statistic=dict(
                key_name="gstlal_far",
                parameter=data["gstlal_far"],
                threshold=1.0,  # FAR < 1 considered detected
            ),
            projected_detection_statistic=dict(
                key_name="observed_snr_net",
                parameter=data["observed_snr_net"],
                threshold=None,
                threshold_search_bounds=(4.0, 14.0),
            ),
            parameters_to_fit=dict(
                key_name="z",
                parameter=data["z"],
            ),
            sample_size=2000,
        )

        # Small number of iterations to keep runtime low in CI
        best_thr, del_H, H, H_true, snr_thrs = finder.find_threshold(
            iteration=6, print_output=False, no_multiprocessing=True
        )

        # Basic validations
        assert np.isfinite(best_thr), "best_thr must be finite"
        assert 4.0 <= best_thr <= 14.0, "best_thr must lie within search bounds"
        assert del_H.shape == (6,) and H.shape == (6,) and H_true.shape == (6,)
        assert snr_thrs.shape == (6,)

        # Numerical sanity: at least some entries should be finite (KDE defined)
        assert np.any(np.isfinite(del_H)), "At least one del_H must be finite"
        assert np.any(np.isfinite(H)), "At least one H must be finite"
        assert np.any(np.isfinite(H_true)), "At least one H_true must be finite"

    def test_find_best_threshold_known_shape(self):
        """find_best_SNR_threshold should recover the maximizer for a smooth curve."""
        finder = SNRThresholdFinder(
            catalog_file=None,
            # Provide minimal valid arrays to satisfy constructor
            selection_range=dict(key_name="mass1_source", parameter=np.array([10.0, 20.0]), range=(5.0, 200.0)),
            original_detection_statistic=dict(key_name="gstlal_far", parameter=np.array([0.5, 0.2]), threshold=1.0),
            projected_detection_statistic=dict(key_name="observed_snr_net", parameter=np.array([8.0, 9.0]), threshold=None, threshold_search_bounds=(4.0, 14.0)),
            parameters_to_fit=dict(key_name="z", parameter=np.array([0.2, 0.3])),
        )

        thrs = np.linspace(4.0, 14.0, 50)
        del_H = -1.0 * (thrs - 9.0) ** 2  # maximum at 9.0
        best_thr = finder.find_best_SNR_threshold(thrs, del_H)
        assert np.isclose(best_thr, 9.0, atol=0.25), f"Expected ~9.0, got {best_thr}"

    def test_hdf5_catalog_ingestion(self):
        """find_threshold using a temporary HDF5 catalog with 'events' dataset."""
        # Skip this test entirely if h5py cannot be imported cleanly
        if H5PY_BROKEN:
            pytest.skip("h5py unavailable or broken in test environment")
        import h5py  # type: ignore
        data = _make_synthetic_data(n=400)

        # Create a structured array to mimic catalog 'events'
        dtype = np.dtype([
            ("z", "f8"),
            ("mass1_source", "f8"),
            ("gstlal_far", "f8"),
            ("observed_snr_net", "f8"),
        ])
        events = np.zeros(len(data["z"]), dtype=dtype)
        events["z"] = data["z"]
        events["mass1_source"] = data["mass1_source"]
        events["gstlal_far"] = data["gstlal_far"]
        events["observed_snr_net"] = data["observed_snr_net"]

        with tempfile.TemporaryDirectory() as td:
            catalog_path = os.path.join(td, "injections.hdf")
            with h5py.File(catalog_path, "w") as f:
                f.create_dataset("events", data=events)
                # Optional: attrs if needed later

            finder = SNRThresholdFinder(
                catalog_file=catalog_path,
                npool=2,
                multiprocessing_verbose=False,
                selection_range=dict(key_name="mass1_source", range=(5.0, 200.0)),
                original_detection_statistic=dict(key_name="gstlal_far", threshold=1.0),
                projected_detection_statistic=dict(key_name="observed_snr_net", threshold=None, threshold_search_bounds=(4.0, 14.0)),
                parameters_to_fit=dict(key_name="z"),
                sample_size=2000,
            )

            best_thr, del_H, H, H_true, snr_thrs = finder.find_threshold(
                iteration=5, print_output=False, no_multiprocessing=True
            )

            assert np.isfinite(best_thr)
            assert 4.0 <= best_thr <= 14.0
            assert del_H.shape == (5,)
            assert snr_thrs.shape == (5,)

    def test_missing_parameters_raises(self):
        """When catalog_file is None, all required arrays must be provided."""
        # Provide only selection_range; omit required arrays for other dicts
        with pytest.raises((ValueError, KeyError)):
            _ = SNRThresholdFinder(
                catalog_file=None,
                selection_range=dict(key_name="mass1_source", parameter=np.array([10.0, 20.0]), range=(5.0, 200.0)),
                # Missing original_detection_statistic['parameter']
                original_detection_statistic=dict(key_name="gstlal_far", threshold=1.0),
                projected_detection_statistic=dict(key_name="observed_snr_net", parameter=np.array([8.0, 9.0]), threshold=None, threshold_search_bounds=(4.0, 14.0)),
                parameters_to_fit=dict(key_name="z", parameter=np.array([0.2, 0.3])),
            )
