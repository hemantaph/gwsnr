"""Regression tests for the full detector norm, including mixed polarizations."""

import numpy as np
import pytest
import bilby
from scipy.interpolate import interp1d

from gwsnr import GWSNR
from gwsnr.numba import antenna_response_array
from gwsnr.utils import multiprocessing_routine as workers


PARAMETERS = {
    "mass_1": 39.39697565794667,
    "mass_2": 38.502200779605765,
    "a_1": 0.013933539741033549,
    "a_2": 0.10387844924090532,
    "tilt_1": 0.3683219347284615,
    "tilt_2": 1.2773472444692886,
    "phi_12": 5.512838537731936,
    "phi_jl": 0.2848074437530412,
    "luminosity_distance": 696.8177279507435,
    "theta_jn": 0.8278270206864476,
    "psi": 2.583807400988801,
    "phase": 0.09150141867219445,
    "geocent_time": 1252983658.1262412,
    "ra": 5.816668997978307,
    "dec": 0.179951954224289,
}
PARAMETER_ORDER = (
    "mass_1", "mass_2", "luminosity_distance", "theta_jn", "psi", "phase",
    "ra", "dec", "geocent_time", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "lambda_1", "lambda_2", "eccentricity",
)
ASD_FILES = {
    "L1": "aLIGO_O4_high_asd.txt",
    "H1": "aLIGO_O4_high_asd.txt",
    "V1": "AdV_asd.txt",
}


@pytest.fixture
def worker_case(monkeypatch):
    """Known complex signals with a nonzero real and imaginary mixed product."""
    frequencies = np.arange(5.0)
    hp = np.array([0, 1 + 2j, -3 + 1j, 2 - 1j, 4j])
    hc = np.array([0, -2 + 1j, 1 + 3j, -1 - 2j, 2 + 1j])
    psd = np.array([1, 2, 3, 4, 5.0])
    psds = [[frequencies, psd, interp1d(frequencies, psd)]]

    class WaveformGenerator:
        def __init__(self, **kwargs):
            self.frequency_array = frequencies
            self.duration = kwargs["duration"]

        def frequency_domain_strain(self, parameters):
            return {"plus": hp, "cross": hc}

    monkeypatch.setattr(bilby.gw, "WaveformGenerator", WaveformGenerator)
    # Restore globals modified by legacy workers after each test.
    monkeypatch.setattr(bilby.core.utils.logger, "disabled", False)
    monkeypatch.setattr(workers, "_worker_shared_data", {
        "psd_list": psds, "approximant": "IMRPhenomXPHM", "f_min": 1.0,
        "f_ref": 1.0, "sampling_frequency": 8.0,
        "frequency_domain_source_model": "lal_binary_black_hole",
    })
    intrinsic = [PARAMETERS.get(key, 0.0) for key in PARAMETER_ORDER]
    slim = (*intrinsic, 1.0, 7)
    legacy = (*intrinsic, "IMRPhenomXPHM", 1.0, 1.0, 1.0, 8.0,
              7, psds, "lal_binary_black_hole")
    ripple = (hp, hc, frequencies, len(frequencies), 1.0, 1.0, 7, psds)
    return hp, hc, psd, {
        "slim": (workers.noise_weighted_inner_prod_h_inner_h_slim, slim),
        "legacy": (workers.noise_weighted_inner_prod_h_inner_h, legacy),
        "noise": (workers.noise_weighted_inner_prod_d_inner_h,
                  (*legacy, np.ones(len(frequencies)))),
        "ripple": (workers.noise_weighted_inner_prod_ripple, ripple),
    }


@pytest.mark.parametrize("worker", ["slim", "legacy", "noise", "ripple"])
def test_worker_products_reconstruct_detector_norm(worker_case, worker):
    hp, hc, psd, cases = worker_case
    function, arguments = cases[worker]
    result = function(arguments)
    pp, cc, pc = [products[0] for products in result[:3]]
    assert result[-1] == 7
    np.testing.assert_allclose(pc, 4 * np.sum(hp.conj() * hc / psd), rtol=1e-14)
    assert abs(pc.real) > 0 and abs(pc.imag) > 0
    # Include either sign, a single polarization, and cancellation between terms.
    for fp, fc in [(0.7, -0.3), (0.7, 0.3), (1.0, 0.0), (0.0, 1.0)]:
        strain = fp * hp + fc * hc
        expected = 4 * np.sum(np.abs(strain)**2 / psd)
        actual = fp**2 * pp + fc**2 * cc + 2 * fp * fc * pc.real
        np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize("worker", ["slim", "legacy", "noise"])
def test_failed_waveform_returns_zero_mixed_products(worker_case, monkeypatch, worker):
    def fail(self, parameters):
        raise RuntimeError("waveform domain error")

    monkeypatch.setattr(bilby.gw.WaveformGenerator, "frequency_domain_strain", fail)
    monkeypatch.setattr(workers, "_warned_waveform_domain_error", False)
    function, arguments = worker_case[-1][worker]
    result = function(arguments)
    assert len(result) == (6 if worker == "noise" else 4)
    assert result[-1] == 7
    np.testing.assert_array_equal(result[:-1], 0.0)


@pytest.mark.parametrize("approximant", ["IMRPhenomD", "IMRPhenomXPHM"])
@pytest.mark.parametrize("verbose,npool", [(False, 1), (True, 2)])
def test_actual_api_matches_bilby_detector_strain(tmp_path, approximant, verbose, npool):
    """Exercise both pool collectors, event indexing, and the network norm."""
    parameters = PARAMETERS.copy()
    if approximant == "IMRPhenomD":
        parameters.update(tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
    # Different distances and polarization angles distinguish events in a batch.
    second = {**parameters, "luminosity_distance": 900.0, "psi": 0.4}
    events = [parameters, second]
    batch = {key: np.array([p[key] for p in events]) for key in parameters}
    calculator = GWSNR(
        snr_method="inner_product", npool=npool,
        waveform_approximant=approximant,
        fixed_duration=64.0, duration_max=64.0,
        sampling_frequency=1024.0, minimum_frequency=20.0, reference_frequency=20.0,
        psds=ASD_FILES.copy(), ifos=list(ASD_FILES),
        interpolator_dir=str(tmp_path),
        gwsnr_verbose=False, multiprocessing_verbose=verbose,
    )
    actual = calculator.optimal_snr(gw_param_dict=batch)
    generator = bilby.gw.WaveformGenerator(
        duration=64.0, sampling_frequency=1024.0,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(waveform_approximant=approximant,
                                minimum_frequency=20.0, reference_frequency=20.0),
    )
    ifos = bilby.gw.detector.InterferometerList(list(ASD_FILES))
    fp, fc = antenna_response_array(
        batch["ra"], batch["dec"], batch["geocent_time"], batch["psi"],
        np.array(calculator.detector_tensor_list),
    )
    for event_index, event in enumerate(events):
        polarizations = generator.frequency_domain_strain(event)
        expected_network2 = 0.0
        for detector_index, ifo in enumerate(ifos):
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                asd_file=ASD_FILES[ifo.name]
            )
            ifo.minimum_frequency, ifo.maximum_frequency = 20.0, 512.0
            ifo.set_strain_data_from_zero_noise(
                sampling_frequency=1024.0, duration=64.0,
                start_time=event["geocent_time"] - 32.0,
            )
            bilby_strain = ifo.get_detector_response(polarizations, event)
            expected2 = np.real(ifo.optimal_snr_squared(bilby_strain))
            snr = actual[f"optimal_snr_{ifo.name}"][event_index]
            # Allow the existing ~1e-8 antenna-factor approximation difference.
            np.testing.assert_allclose(snr, np.sqrt(expected2), rtol=1e-7)
            # With identical antenna factors, require near machine precision.
            projected = (fp[detector_index, event_index] * polarizations["plus"]
                         + fc[detector_index, event_index] * polarizations["cross"])
            projected2 = np.real(ifo.optimal_snr_squared(projected))
            np.testing.assert_allclose(snr**2, projected2, rtol=1e-12)
            expected_network2 += expected2
        np.testing.assert_allclose(actual["optimal_snr_net"][event_index],
                                   np.sqrt(expected_network2), rtol=1e-7)


def test_recalculation_and_pdet_use_corrected_inner_product(tmp_path):
    """Downstream dispatch must preserve the corrected precessing SNR."""
    calculator = GWSNR(
        snr_method="inner_product", waveform_approximant="IMRPhenomXPHM",
        npool=1, fixed_duration=64.0, duration_max=64.0,
        sampling_frequency=1024.0, minimum_frequency=20.0, reference_frequency=20.0,
        psds=ASD_FILES.copy(), interpolator_dir=str(tmp_path),
        gwsnr_verbose=False, multiprocessing_verbose=False,
    )
    batch = {key: np.array([value]) for key, value in PARAMETERS.items()}
    expected = calculator.optimal_snr(gw_param_dict=batch)
    calculator.snr_recalculation = True
    calculator.snr_recalculation_range = [0.0, 100.0]
    calculator.snr_recalculation_waveform_approximant = "IMRPhenomXPHM"
    recalculated = calculator.optimal_snr(gw_param_dict=batch)
    detected = calculator.pdet(
        gw_param_dict=batch, pdet_type="boolean", distribution_type="fixed_snr",
        include_optimal_snr=True, snr_th_net=50.1,
    )
    for key, value in expected.items():
        np.testing.assert_allclose(recalculated[key], value, rtol=1e-12)
        np.testing.assert_allclose(detected[key], value, rtol=1e-12)
    # The old two-term network SNR (50.1809) would incorrectly exceed this threshold.
    np.testing.assert_array_equal(detected["pdet_net"], [0])
