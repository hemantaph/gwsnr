import numpy as np
from gwsnr import GWSNR
import pytest

def test_gwsnr_snr_generation(tmp_path):
    # Initialize the GWSNR object
    gwsnr = GWSNR(
        npool=4,
        mtot_min=2.0,
        mtot_max=439.6,
        ratio_min=0.1,
        ratio_max=1.0,
        mtot_resolution=100,
        ratio_resolution=20,
        sampling_frequency=2048.0,
        waveform_approximant="IMRPhenomD",
        minimum_frequency=20.0,
        snr_type="interpolation",
        psds=None,
        ifos=None,
        interpolator_dir="./interpolator_pickle",
        create_new_interpolator=False,
        gwsnr_verbose=False,
        multiprocessing_verbose=False,
        mtot_cut=True,
    )

    # Generate BBH parameters
    nsamples = 10
    chirp_mass = np.linspace(5, 80, nsamples)
    mass_ratio = np.random.uniform(0.2, 1, size=nsamples)
    mass_1 = (chirp_mass * (1 + mass_ratio) ** (1 / 5)) / mass_ratio ** (3 / 5)
    mass_2 = chirp_mass * mass_ratio ** (2 / 5) * (1 + mass_ratio) ** (1 / 5)
    total_mass = mass_1 + mass_2
    mass_ratio = mass_2 / mass_1
    luminosity_distance = 80 * np.ones(nsamples)
    theta_jn = np.random.uniform(0, 2 * np.pi, size=nsamples)
    ra = np.random.uniform(0, 2 * np.pi, size=nsamples)
    dec = np.random.uniform(0, np.pi, size=nsamples)
    psi = np.random.uniform(0, 2 * np.pi, size=nsamples)
    phase = np.random.uniform(0, 2 * np.pi, size=nsamples)

    # Calculate SNR
    interp_snr = gwsnr.snr(
        mass_1=mass_1,
        mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        theta_jn=theta_jn,
        psi=psi,
        phase=phase,
        ra=ra,
        dec=dec,
    )

    # Assertions to verify output
    assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
    assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"
    assert len(interp_snr["optimal_snr_net"]) == nsamples, "SNR result length mismatch"

    # Optional: Test saving to JSON (isolated via tmp_path)
    from gwsnr.utils import save_json
    data = {
        'mass_1': mass_1.tolist(),
        'mass_2': mass_2.tolist(),
        'luminosity_distance': luminosity_distance.tolist(),
        'theta_jn': theta_jn.tolist(),
        'psi': psi.tolist(),
        'phase': phase.tolist(),
        'ra': ra.tolist(),
        'dec': dec.tolist()
    }
    data.update({k: v.tolist() for k, v in interp_snr.items()})

    output_file = tmp_path / "snr_data.json"
    save_json(output_file, data)
    assert output_file.exists(), "Output JSON file was not created"
