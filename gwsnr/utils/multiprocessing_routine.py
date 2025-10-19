# -*- coding: utf-8 -*-
"""
Helper functions for multiprocessing in snr generation
"""

import numpy as np
import bilby

from ..numba import noise_weighted_inner_product


def noise_weighted_inner_prod_h_inner_h(params):
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
            lambda_1
        params[16] : float
            lambda_2
        params[17] : float
            eccentricity
        params[18] : float
            approximant
        params[19] : float
            f_min
        params[20] : float
            f_ref
        params[21] : float
            duration
        params[22] : float
            sampling_frequency
        params[23] : int
            index tracker
        params[24] : list
            list of psds for each detector
        params[25] : str
            frequency_domain_source_model name
        

    Returns
    -------
    SNRs_list : list
        contains opt_snr for each detector and net_opt_snr
    params[22] : int
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
        "lambda_1": params[15],
        "lambda_2": params[16],
        "eccentricity": params[17],
    }

    # print('eccentricity', params[17])
    # print('frequency_domain_source_model', params[25])

    waveform_arguments = dict(
        waveform_approximant=params[18],
        reference_frequency=params[20],
        minimum_frequency=params[19],
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=params[21],
        sampling_frequency=params[22],
        frequency_domain_source_model=getattr(bilby.gw.source, params[25]),
        waveform_arguments=waveform_arguments,
    )
    polas = waveform_generator.frequency_domain_strain(parameters=parameters)

    # h = F+.h+ + Fx.hx
    # <h|h> = F+^2<h+,h+> + Fx^2<hx,hx> + F+Fx 2<h+,hx>
    # <h|h> = F+^2<h+,h+> + Fx^2<hx,hx>, if h+ and hx are orthogonal
    # F+^2 and Fx^2 will be added later
    hp_inner_hp_list = []
    hc_inner_hc_list = []
    # list_of_detectors = params[26:].tolist()
    psds_objects = params[24]
    for idx in range(len(psds_objects)):
    # for idx, det in enumerate(list_of_detectors):

        # need to compute the inner product for
        p_array = psds_objects[idx][2](waveform_generator.frequency_array)
        # p_array =  cubic_spline_interpolator(xnew_array=waveform_generator.frequency_array, coefficients=psds_objects[idx][2], x=psds_objects[idx][0])

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

        hp_inner_hp_list.append(hp_inner_hp)
        hc_inner_hc_list.append(hc_inner_hc)

    return (hp_inner_hp_list, hc_inner_hc_list, params[23])

def noise_weighted_inner_prod_d_inner_h(params):
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
            lambda_1
        params[16] : float
            lambda_2
        params[17] : float
            eccentricity
        params[18] : float
            approximant
        params[19] : float
            f_min
        params[20] : float
            f_ref
        params[21] : float
            duration
        params[22] : float
            sampling_frequency
        params[23] : int
            index tracker
        params[24] : list
            list of psds for each detector
        params[25] : str
            frequency_domain_source_model name
        params[26] : list or None
            noise realization. If None, then PSD as noise realization
        

    Returns
    -------
    SNRs_list : list
        contains opt_snr for each detector and net_opt_snr
    params[22] : int
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
        "lambda_1": params[15],
        "lambda_2": params[16],
        "eccentricity": params[17],
    }

    # print('eccentricity', params[17])
    # print('frequency_domain_source_model', params[25])

    waveform_arguments = dict(
        waveform_approximant=params[18],
        reference_frequency=params[20],
        minimum_frequency=params[19],
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=params[21],
        sampling_frequency=params[22],
        frequency_domain_source_model=getattr(bilby.gw.source, params[25]),
        waveform_arguments=waveform_arguments,
    )
    polas = waveform_generator.frequency_domain_strain(parameters=parameters)

    # h = F+.h+ + Fx.hx
    # <h|h> = F+^2<h+,h+> + Fx^2<hx,hx> + F+Fx 2<h+,hx>
    # <h|h> = F+^2<h+,h+> + Fx^2<hx,hx>, if h+ and hx are orthogonal
    # F+^2 and Fx^2 will be added later
    hp_inner_hp_list = []
    hc_inner_hc_list = []
    n_inner_hp_list = []
    n_inner_hc_list = []
    # list_of_detectors = params[26:].tolist()
    psds_objects = params[24]
    noise = params[26]
    for idx in range(len(psds_objects)):
    # for idx, det in enumerate(list_of_detectors):

        # need to compute the inner product for
        p_array = psds_objects[idx][2](waveform_generator.frequency_array)
        if noise is None:
            noise = np.sqrt(p_array)
        else:
            noise = np.array(noise)

        idx2 = (noise != 0.0) & (noise != np.inf)
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
        n_inner_hp = noise_weighted_inner_product(
            noise[idx2],
            polas["plus"][idx2],
            p_array[idx2],
            waveform_generator.duration,
        )
        n_inner_hc = noise_weighted_inner_product(
            noise[idx2],
            polas["cross"][idx2],
            p_array[idx2],
            waveform_generator.duration,
        )
        # if idx == 2:
        #     # save n_inner_hp and n_inner_hc
        #     np.save("n_inner_hp.npy", n_inner_hp)
        #     np.save("n_inner_hc.npy", n_inner_hc)
        #     np.save("noise.npy", noise[idx2])
        #     np.save("p_array.npy", p_array[idx2])
        #     np.save("polas_plus.npy", polas["plus"][idx2])
        #     np.save("polas_cross.npy", polas["cross"][idx2])
        #     np.save("duration.npy", waveform_generator.duration)

        hp_inner_hp_list.append(hp_inner_hp)
        hc_inner_hc_list.append(hc_inner_hc)
        n_inner_hp_list.append(n_inner_hp)
        n_inner_hc_list.append(n_inner_hc)

    return (hp_inner_hp_list, hc_inner_hc_list, n_inner_hp_list, n_inner_hc_list, params[23])

def noise_weighted_inner_prod_ripple(params):
    """
        Probaility of detection of GW for the given sensitivity of the detectors

        Parameters
        ----------
        params : list
            list of parameters for the inner product calculation
            List contains: \n
            params[0] : `numpy.ndarray`
                plus polarization
            params[1] : `numpy.ndarray`
                cross polarization
            params[2] : `numpy.ndarray`
                frequency array
            params[3] : `float`
                cutt-off size of given arrays
            params[4] : `float`
                minimum frequency
            params[5] : `float`
                duration
            params[6] : `int`
                index
            params[7] : `list` 
                psd objects of given detectors

        Returns
        -------
        SNRs_list : list
            contains opt_snr for each detector and net_opt_snr
        params[22] : int
            index tracker
    """

    ## input 
    hp = params[0]
    hc = params[1]
    fs = params[2]
    fsize = params[3]
    fmin = params[4]
    duration = params[5]

    # for i in range(size):
    # remove the np.nan padding
    hp = np.array(hp[:fsize])
    hc = np.array(hc[:fsize])
    # find the index of 20Hz or nearby
    # set all elements to zero below this index
    fs =  fs[:fsize]
    idx = np.abs( fs - fmin).argmin()
    hp[:idx] = 0.0 + 0.0j
    hc[:idx] = 0.0 + 0.0j

    # h = F+.h+ + Fx.hx
    # <h|h> = <h+,h+> + <hx,hx> + 2<h+,hx>
    # <h|h> = <h+,h+> + <hx,hx>, if h+ and hx are orthogonal
    hp_inner_hp_list = []
    hc_inner_hc_list = []
    psds_objects = params[7]
    
    for idx in range(len(psds_objects)):

        # need to compute the inner product for
        p_array = psds_objects[idx][2](fs)

        idx2 = (p_array != 0.0) & (p_array != np.inf)
        # complex128 is not available in JAX numpy operation.
        # That's why I will use numba njit function
        hp_inner_hp = noise_weighted_inner_product(
            hp[idx2],
            hp[idx2],
            p_array[idx2],
            duration,
        )
        # complex128 is not available in JAX numpy operation.
        # That's why I will use numba njit function
        hc_inner_hc = noise_weighted_inner_product(
            hc[idx2],
            hc[idx2],
            p_array[idx2],
            duration,
        )

        hp_inner_hp_list.append(hp_inner_hp)
        hc_inner_hc_list.append(hc_inner_hc)
    
    return (hp_inner_hp_list, hc_inner_hc_list, params[6])


