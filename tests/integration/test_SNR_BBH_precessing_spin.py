'''
Generate SNR for 10 random BBH events (with precessing spin) and save it to a json file.
Use detectors L1, H1, V1 with O4 design sensitivity
'''

import numpy as np
from gwsnr import GWSNR
from gwsnr.utils import append_json
import pytest
np.random.seed(1234)

def test_gwsnr_snr_generation(tmp_path):
    # Initialize the GWSNR object
    gwsnr = GWSNR(
        npool=4,
        waveform_approximant="IMRPhenomXPHM",
        snr_type="inner_product",
        ifos=["L1", "H1", "V1"],
        interpolator_dir="./interpolator_pickle",
        create_new_interpolator=False,
        gwsnr_verbose=False,
        multiprocessing_verbose=False,
    )

    # Generate BBH parameters
    nsamples = 10
    chirp_mass = np.linspace(5,60,nsamples)
    mass_ratio = np.random.uniform(0.2,1,size=nsamples)
    param_dict = dict(
        mass_1 = (chirp_mass*(1+mass_ratio)**(1/5))/mass_ratio**(3/5),
        mass_2 = chirp_mass*mass_ratio**(2/5)*(1+mass_ratio)**(1/5),
        # Fix luminosity distance
        luminosity_distance = 500*np.ones(nsamples),
        # Randomly sample everything else:
        theta_jn = np.random.uniform(0,2*np.pi, size=nsamples),
        ra = np.random.uniform(0,2*np.pi, size=nsamples), 
        dec = np.random.uniform(0,np.pi, size=nsamples), 
        psi = np.random.uniform(0,2*np.pi, size=nsamples),
        phase = np.random.uniform(0,2*np.pi, size=nsamples),
        geocent_time = 1246527224.169434*np.ones(nsamples),
        # spin non-zero
        a_1 = np.random.uniform(0,0.8, size=nsamples),
        a_2 = np.random.uniform(0,0.8, size=nsamples),
        tilt_1 = np.random.uniform(0,np.pi, size=nsamples),
        tilt_2 = np.random.uniform(0,np.pi, size=nsamples),
        phi_12 = np.random.uniform(0,2*np.pi, size=nsamples),
        phi_jl = np.random.uniform(0,2*np.pi, size=nsamples),
    )

    # Calculate SNR
    interp_snr = gwsnr.snr(
        gw_param_dict=param_dict
    )

    # Assertions to verify output
    assert isinstance(interp_snr, dict), "SNR output should be a dictionary"
    assert "optimal_snr_net" in interp_snr, "Expected 'optimal_snr_net' in SNR output"
    assert len(interp_snr["optimal_snr_net"]) == nsamples, "SNR result length mismatch"

    # Optional: Test saving to JSON (isolated via tmp_path)
    param_dict.update(interp_snr)
    output_file = tmp_path / "BBH_precessing_spin.json"
    append_json(output_file, param_dict, replace=True)
    assert output_file.exists(), "Output JSON file was not created"