# -*- coding: utf-8 -*-
"""
Helper functions for multiprocessing in snr generation
"""

import numpy as np
import bilby

from .njit_functions import noise_weighted_inner_product


def noise_weighted_inner_prod(params):
    """
    Probaility of detection of GW for the given sensitivity of the detectors

    Parameters
    ----------
    params : list
        list of parameters for the inner product calculation
        List contains: \n
        params[0] : float
            mass_1
        params[1] : float
            mass_2
        params[2] : float
            luminosity_distance
        params[3] : float
            theta_jn
        params[4] : float
            psi
        params[5] : float
            phase
        params[6] : float
            ra
        params[7] : float
            dec
        params[8] : float
            geocent_time
        params[9] : float
            a_1
        params[10] : float
            a_2
        params[11] : float
            tilt_1
        params[12] : float
            tilt_2
        params[13] : float
            phi_12
        params[14] : float
            phi_jl
        params[15] : float
            approximant
        params[16] : float
            f_min
        params[17] : float
            duration
        params[18] : float
            sampling_frequency
        params[19] : int
            index tracker
        psds_list[20] : list
            list of psds for each detector
        detector_list[21:] : list
            list of detectors

    Returns
    -------
    SNRs_list : list
        contains opt_snr for each detector and net_opt_snr
    params[19] : int
        index tracker
    """

    bilby.core.utils.logger.disabled = True
    np.random.seed(88170235)
    parameters = {
        "mass_1": params[0],
        "mass_2": params[1],
        "luminosity_distance": params[2],
        "theta_jn": params[3],
        "psi": params[4],
        "phase": params[5],
        "geocent_time": params[8],
        "ra": params[6],
        "dec": params[7],
        "a_1": params[9],
        "a_2": params[10],
        "tilt_1": params[11],
        "tilt_2": params[12],
        "phi_12": params[13],
        "phi_jl": params[14],
    }

    waveform_arguments = dict(
        waveform_approximant=params[15],
        reference_frequency=20.0,
        minimum_frequency=params[16],
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=params[17],
        sampling_frequency=params[18],
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
    )
    polas = waveform_generator.frequency_domain_strain(parameters=parameters)

    # h = F+.h+ + Fx.hx
    # <h|h> = <h+,h+> + <hx,hx> + 2<h+,hx>
    # <h|h> = <h+,h+> + <hx,hx>, if h+ and hx are orthogonal
    hp_inner_hp_list = []
    hc_inner_hc_list = []
    list_of_detectors = params[21:].tolist()
    psds_objects = params[20]
    for det in list_of_detectors:

        # need to compute the inner product for
        p_array = psds_objects[det].get_power_spectral_density_array(waveform_generator.frequency_array)
        idx2 = (p_array != 0.0) & (p_array != np.inf)
        hp_inner_hp = noise_weighted_inner_product(
            polas["plus"][idx2],
            polas["plus"][idx2],
            p_array[idx2],
            waveform_generator.duration,
        )
        hc_inner_hc = noise_weighted_inner_product(
            polas["cross"][idx2],
            polas["cross"][idx2],
            p_array[idx2],
            waveform_generator.duration,
        )

        # might need to add these lines in the future for waveform with multiple harmonics and h+ and hx are not orthogonal
        # hp_inner_hc = bilby.gw.utils.noise_weighted_inner_product(
        #     polas["plus"][idx2],
        #     polas["cross"][idx2],
        #     p_array[idx2],
        #     waveform_generator.duration,
        # )

        hp_inner_hp_list.append(hp_inner_hp)
        hc_inner_hc_list.append(hc_inner_hc)

    return (hp_inner_hp_list, hc_inner_hc_list, params[19])