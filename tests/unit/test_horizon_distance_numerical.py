from types import SimpleNamespace

import numpy as np

from gwsnr.core.gwsnr import GWSNR


def _minimal_gwsnr(detectors=None):
    gwsnr = object.__new__(GWSNR)
    gwsnr.detector_list = list(detectors or [])
    gwsnr.detector_tensor_list = [np.eye(3) for _ in gwsnr.detector_list]
    gwsnr.pdet_kwargs = {"snr_th": 5.0, "snr_th_net": 11.0}
    gwsnr.multiprocessing_verbose = True
    return gwsnr


def test_horizon_distance_numerical_appends_inclination_bound(monkeypatch):
    gwsnr = _minimal_gwsnr()
    optimal_snr_calls = []

    def optimal_snr(**kwargs):
        optimal_snr_calls.append(kwargs)
        return {"optimal_snr_net": np.array([22.0])}

    gwsnr.optimal_snr = optimal_snr

    def differential_evolution(func, bounds, tol, polish, maxiter):
        assert bounds == [(0, 2 * np.pi), (-np.pi / 2, np.pi / 2), (0, np.pi)]
        x = np.array([1.1, 0.2, 0.7])
        assert np.isfinite(func(x))
        return SimpleNamespace(x=x)

    def root_scalar(func, bracket, method, xtol):
        assert func(100.0) == 11.0
        return SimpleNamespace(root=100.0)

    monkeypatch.setattr(
        "scipy.optimize.differential_evolution",
        differential_evolution,
    )
    monkeypatch.setattr("scipy.optimize.root_scalar", root_scalar)

    horizon, optimal_orientation = gwsnr.horizon_distance_numerical(
        mass_1=30.0,
        mass_2=30.0,
        minimize_function_dict={
            "bounds": [(0, 2 * np.pi), (-np.pi / 2, np.pi / 2)],
            "tol": 1e-7,
            "polish": True,
            "maxiter": 10000,
        },
    )

    assert horizon["horizon_distance_net"] == 100.0
    assert optimal_orientation["optimal_orientation_net"] == {
        "ra": 1.1,
        "dec": 0.2,
        "theta_jn": 0.7,
    }
    assert optimal_snr_calls[0]["theta_jn"] == 0.7
    assert optimal_snr_calls[-1]["theta_jn"] == 0.7


def test_detector_location_as_optimal_sky_sets_detector_face_on(monkeypatch):
    gwsnr = _minimal_gwsnr(["H1"])
    gwsnr.ifos = [
        SimpleNamespace(latitude=46.455, longitude=-119.408, elevation=142.0)
    ]
    optimal_snr_calls = []

    def optimal_snr(**kwargs):
        optimal_snr_calls.append(kwargs)
        return {
            "optimal_snr_H1": np.array([20.0]),
            "optimal_snr_net": np.array([24.0]),
        }

    gwsnr.optimal_snr = optimal_snr

    def differential_evolution(func, bounds, tol, polish, maxiter):
        x = np.array([2.0, 0.1, 1.3])
        assert np.isfinite(func(x))
        return SimpleNamespace(x=x)

    def root_scalar(func, bracket, method, xtol):
        func(100.0)
        return SimpleNamespace(root=100.0)

    monkeypatch.setattr(
        "scipy.optimize.differential_evolution",
        differential_evolution,
    )
    monkeypatch.setattr("scipy.optimize.root_scalar", root_scalar)

    _, optimal_orientation = gwsnr.horizon_distance_numerical(
        mass_1=30.0,
        mass_2=30.0,
        theta_jn=2.4,
        detector_location_as_optimal_sky=True,
    )

    root_calls = [
        call for call in optimal_snr_calls if call["luminosity_distance"] == 100.0
    ]
    assert root_calls[0]["theta_jn"] == 0.0
    assert root_calls[1]["theta_jn"] == 1.3
    assert optimal_orientation["optimal_orientation_H1"]["theta_jn"] == 0.0
    assert optimal_orientation["optimal_orientation_net"] == {
        "ra": 2.0,
        "dec": 0.1,
        "theta_jn": 1.3,
    }
