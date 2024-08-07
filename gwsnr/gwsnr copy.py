# -*- coding: utf-8 -*-
"""
This module contains functions for calculating the SNR of a CBC signal. It has two methods: interpolation (bicubic) and inner product. Interpolation method is much faster than inner product method. Interpolation method is tested for IMRPhenomD and TaylorF2 waveform approximants for the spinless scenario.
"""

import os
import pickle
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from tensorflow.keras.models import load_model

from .utils import (
    save_json,
    dealing_with_psds,
    interpolator_check,
    load_json,
    save_json_dict,
    load_h5_from_module,
    load_pkl_from_module,
)
from .njit_functions import (
    get_interpolated_snr,
    findchirp_chirptime,
    antenna_response,
    antenna_response_array,
)
from .multiprocessing_routine import noise_weighted_inner_prod

# defining constants
C = 299792458.0
G = 6.67408e-11
Pi = np.pi
MTSUN_SI = 4.925491025543576e-06


class GWSNR:
    """
    Class to calculate SNR of a CBC signal with either interpolation or inner product method. Interpolation method is much faster than inner product method. Interpolation method is tested for IMRPhenomD and TaylorF2 waveform approximants for the spinless scenario.

    Parameters
    ----------
    npool : `int`
        Number of processors to use for parallel processing.
        Default is 4.
    mtot_min : `float`
        Minimum total mass of the binary in solar mass (use interpolation purpose). Default is 2.0.
    mtot_max : `float`
        Maximum total mass of the binary in solar mass (use interpolation purpose). Default is 184. This is set so that the waveform is within the frequency range of the detector (with fmin=20.).
    ratio_min : `float`
        Minimum mass ratio of the binary (use interpolation purpose). Default is 0.1.
    ratio_max : `float`
        Maximum mass ratio of the binary (use interpolation purpose). Default is 1.0.
    mtot_resolution : `int`
        Number of points in the total mass array (use interpolation purpose). Default is 100.
    ratio_resolution : `int`
        Number of points in the mass ratio array (use interpolation purpose). Default is 100.
    sampling_frequency : `float`
        Sampling frequency of the detector. Default is 2048.0.
    waveform_approximant : `str`
        Waveform approximant to use. Default is 'IMRPhenomD'.
    minimum_frequency : `float`
        Minimum frequency of the waveform. Default is 20.0.
    snr_type : `str`
        Type of SNR calculation. Default is 'interpolation'.
        options: 'interpolation', 'inner_product', 'pdet', 'ann'
    psds : `dict`
        Dictionary of psds for different detectors. Default is None. If None, bilby's default psds will be used. Other options:\n
        Example 1: when values are psd name from pycbc analytical psds, psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044','V1':'AdvVirgo'}. To check available psd name run \n
        >>> import pycbc.psd
        >>> pycbc.psd.get_lalsim_psd_list()
        Example 2: when values are psd txt file available in bilby,
        psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt', 'K1':'KAGRA_design_asd.txt'}.
        For other psd files, check https://github.com/lscsoft/bilby/tree/master/bilby/gw/detector/noise_curves \n
        Example 3: when values are custom psd txt file. psds={'L1':'custom_psd.txt','H1':'custom_psd.txt'}. Custom created txt file has two columns. 1st column: frequency array, 2nd column: strain.
    ifos : `list` or `None`
        List of interferometer objects or detector names. Default is None. If None, bilby's default interferometer objects will be used. For example for LIGO India detector, it can be defined as follows, \n
        >>> import bilby
        >>> from gwsnr import GWSNR
        >>> ifosLIO = bilby.gw.detector.interferometer.Interferometer(
                name = 'LIO',
                power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file='your_asd_file.txt'),
                minimum_frequency = 10.,
                maximum_frequency = 2048.,
                length = 4,
                latitude = 19 + 36. / 60 + 47.9017 / 3600,
                longitude = 77 + 01. / 60 + 51.0997 / 3600,
                elevation = 450.,
                xarm_azimuth = 117.6157,
                yarm_azimuth = 117.6157 + 90.,
                xarm_tilt = 0.,
                yarm_tilt = 0.)
        >>> snr = GWSNR(psds=dict(LIO='your_asd.txt'), ifos=[ifosLIO])
    interpolator_dir : `str`
        Path to store the interpolator pickle file. Default is './interpolator_pickle'.
    create_new_interpolator : `bool`
        If set True, new interpolator will be generated or replace the existing one. Default is False.
    gwsnr_verbose : `bool`
        If True, print all the parameters of the class instance. Default is True.
    multiprocessing_verbose : `bool`
        If True, it will show progress bar while computing SNR (inner product) with :meth:`~snr_with_interpolation`. Default is True. If False, it will not show progress bar but will be faster.
    mtot_cut : `bool`
        If True, it will set the maximum total mass of the binary according to the minimum frequency of the waveform. Default is True.

    Examples
    ----------
    >>> from gwsnr import GWSNR
    >>> snr = GWSNR()
    >>> snr.snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)

    Instance Attributes
    ----------
    GWSNR class has the following attributes, \n
    +-------------------------------------+----------------------------------+
    | Atrributes                          | Type                             |
    +=====================================+==================================+
    |:attr:`~npool`                       | `int`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~mtot_min`                    | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~mtot_max`                    | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~ratio_min`                   | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~ratio_max`                   | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~mtot_resolution`             | `int`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~ratio_resolution`            | `int`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~ratio_arr`                   | `numpy.ndarray`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~mtot_arr`                    | `numpy.ndarray`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~sampling_frequency`          | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~waveform_approximant`        | `str`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~f_min`                       | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~snr_type`                    | `str`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~interpolator_dir`            | `str`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~psds_list`                   | `list` of bilby's                |
    |                                     |  PowerSpectralDensity `object`   |
    +-------------------------------------+----------------------------------+
    |:attr:`~detector_tensor_list`        | `list` of detector tensor        |
    |                                     |  `numpy.ndarray`                 |
    +-------------------------------------+----------------------------------+
    |:attr:`~detector_list`               | `list` of `str`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~path_interpolator`           | `list` of `str`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~multiprocessing_verbose`     | `bool`                           |
    +-------------------------------------+----------------------------------+

    Instance Methods
    ----------
    GWSNR class has the following methods, \n
    +-------------------------------------+----------------------------------+
    | Methods                             | Description                      |
    +=====================================+==================================+
    |:meth:`~snr`                         | Calls                            |
    |                                     | :meth:`~snr_with_interpolation`  |
    |                                     | or :meth:`~compute_bilby_snr`    |
    |                                     | depending on the value of        |
    |                                     | :attr:`~snr_type` attribute.     |
    +-------------------------------------+----------------------------------+
    |:meth:`~snr_with_interpolation`      | Calculates SNR using             |
    |                                     | interpolation method.            |
    +-------------------------------------+----------------------------------+
    |:meth:`~compute_bilby_snr`           | Calculates SNR using             |
    |                                     | inner product method.            |
    +-------------------------------------+----------------------------------+
    |:meth:`~bns_horizon`                 | Calculates BNS horizon           |
    |                                     | distance.                        |
    +-------------------------------------+----------------------------------+
    |:meth:`~print_all_params`            | Prints all the parameters of     |
    |                                     | the class instance.              |
    +-------------------------------------+----------------------------------+
    |:meth:`~init_partialscaled`             | Generates partialscaled SNR         |
    |                                     | interpolation coefficients.      |
    +-------------------------------------+----------------------------------+
    """

    # Attributes
    npool = None
    """``int`` \n
    Number of processors to use for parallel processing."""

    mtot_min = None
    """``float`` \n
    Minimum total mass of the binary in solar mass (use interpolation purpose)."""

    mtot_max = None
    """``float`` \n
    Maximum total mass of the binary in solar mass (use interpolation purpose)."""

    ratio_min = None
    """``float`` \n
    Minimum mass ratio of the binary (use interpolation purpose)."""

    ratio_max = None
    """``float`` \n
    Maximum mass ratio of the binary (use interpolation purpose)."""

    mtot_resolution = None
    """``int`` \n
    Number of points in the total mass array (use interpolation purpose)."""

    ratio_resolution = None
    """``int`` \n
    Number of points in the mass ratio array (use interpolation purpose)."""

    ratio_arr = None
    """``numpy.ndarray`` \n
    Array of mass ratio."""

    snr_partialsacaled = None
    """``numpy.ndarray`` \n
    Array of partial scaled SNR interpolation coefficients."""

    sampling_frequency = None
    """``float`` \n
    Sampling frequency of the detector."""

    waveform_approximant = None
    """``str`` \n
    Waveform approximant to use."""

    f_min = None
    """``float`` \n
    Minimum frequency of the waveform."""

    snr_type = None
    """``str`` \n
    Type of SNR calculation."""

    psds = None
    """``dict`` \n
    Dictionary of psds for different detectors."""

    interpolator_dir = None
    """``str`` \n
    Path to store the interpolator pickle file."""

    detector_list = None
    """``list`` \n
    List of detectors."""

    stored_snrs = None
    """``dict`` \n
    Dictionary of stored SNRs."""

    pdet = None
    """``bool`` \n
    If True, it will calculate the probability of detection. Default is False. Can also be 'matched_filter' or 'bool'. The value 'True' and 'bool' will give the same result."""

    snr_th = None
    """``float`` \n
    SNR threshold for individual detector. Use for pdet calculation. Default is 8.0."""

    snr_th_net = None
    """``float`` \n
    SNR threshold for network SNR. Use for pdet calculation. Default is 8.0."""

    def __init__(
        self,
        npool=int(4),
        mtot_min=2.0,
        mtot_max=439.6,
        ratio_min=0.1,
        ratio_max=1.0,
        mtot_resolution=500,
        ratio_resolution=50,
        sampling_frequency=2048.0,
        waveform_approximant="IMRPhenomD",
        minimum_frequency=20.0,
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
        ann_dir="ann_data/",
    ):
        # setting instance attributes
        self.npool = npool
        self.pdet = pdet
        self.snr_th = snr_th
        self.snr_th_net = snr_th_net

        # dealing with mtot_max
        # set max cut off according to minimum_frequency
        mtot_max = (
            mtot_max
            if not mtot_cut
            else self.calculate_mtot_max(mtot_max, minimum_frequency)
        )
        self.mtot_max = mtot_max
        self.mtot_min = mtot_min
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.mtot_resolution = mtot_resolution
        self.ratio_resolution = ratio_resolution
        # buffer of 0.01 is added to the ratio
        self.ratio_arr = np.geomspace(ratio_min, ratio_max, ratio_resolution)
        # buffer of 0.1 is added to the mtot
        self.mtot_arr = np.sort(
            mtot_min + mtot_max - np.geomspace(mtot_min, mtot_max, mtot_resolution)
        )
        self.sampling_frequency = sampling_frequency
        self.waveform_approximant = waveform_approximant
        self.f_min = minimum_frequency
        self.snr_type = snr_type
        self.interpolator_dir = interpolator_dir

        # dealing with psds
        # if not given, bilby's default psds will be used
        # interferometer object will be created for Fp, Fc calculation
        # self.psds and self.ifos are list of dictionaries
        # self.detector_list are list of strings and will be set at the last.
        psds_list, detector_tensor_list, detector_list = dealing_with_psds(
            psds, ifos, minimum_frequency, sampling_frequency
        )

        # param_dict_given is an identifier for the interpolator
        self.param_dict_given = {
            "mtot_min": self.mtot_min,
            "mtot_max": self.mtot_max,
            "mtot_resolution": self.mtot_resolution,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "ratio_resolution": self.ratio_resolution,
            "sampling_frequency": self.sampling_frequency,
            "waveform_approximant": self.waveform_approximant,
            "minimum_frequency": self.f_min,
            "detector": detector_list,
            "psds": psds_list,
            "detector_tensor": detector_tensor_list,
        }

        # dealing with interpolator
        def interpolator_setup():
            # Note: it will only select detectors that does not have interpolator stored yet
            (
                self.psds_list,
                self.detector_tensor_list,
                self.detector_list,
                self.path_interpolator,
                path_interpolator_all,
            ) = interpolator_check(
                param_dict_given=self.param_dict_given.copy(),
                interpolator_dir=interpolator_dir,
                create_new=create_new_interpolator,
            )

            self.multiprocessing_verbose = False  # This lets multiprocessing to use map instead of imap_unordered function.
            # len(detector_list) == 0, means all the detectors have interpolator stored
            if len(self.detector_list) > 0:
                print("Please be patient while the interpolator is generated")
                self.init_partialscaled()
            elif create_new_interpolator:
                # change back to original
                self.psds_list = psds_list
                self.detector_tensor_list = detector_tensor_list
                self.detector_list = detector_list
                print("Please be patient while the interpolator is generated")
                self.init_partialscaled()

            # get all partialscaledSNR from the stored interpolator
            self.snr_partialsacaled_list = [
                load_json(path) for path in path_interpolator_all
            ]

            return path_interpolator_all

        # now generate interpolator, if not exists
        if snr_type == "interpolation":
            # dealing with interpolator
            path_interpolator_all = interpolator_setup()
        # inner product method doesn't need interpolator generation
        elif snr_type == "inner_product":
            pass
        # ANN method still needs the partialscaledSNR interpolator.
        elif snr_type == "ann":
            self.ann_path_dict = ann_path_dict
            # create dir if not exists
            if not os.path.exists(ann_dir):
                os.makedirs(ann_dir)

            if self.ann_path_dict is None:
                print("You have chosen default ANN model. This model only works for gwsnr default parameters.")
                print("ANN model will be save and loaded from 'ann_data' directory. To create new model, follow instructions from the 'gwsnr' documentation.")
                # load the model
                # save it in ANN directory
                if not os.path.exists(f"{ann_dir}/ann_model_L1_O4.h5"):
                    modelL1 = load_h5_from_module('gwsnr', 'ann', 'ann_model_L1_O4.h5')
                    modelL1.save(f"{ann_dir}/ann_model_L1_O4.h5")
                    del modelL1
                if not os.path.exists(f"{ann_dir}/ann_model_H1_O4.h5"):
                    modelH1 = load_h5_from_module('gwsnr', 'ann', 'ann_model_H1_O4.h5')
                    modelH1.save(f"{ann_dir}/ann_model_H1_O4.h5")
                    del modelH1
                if not os.path.exists(f"{ann_dir}/ann_model_V1_O4.h5"):
                    modelV1 = load_h5_from_module('gwsnr', 'ann', 'ann_model_V1_O4.h5')
                    modelV1.save(f"{ann_dir}/ann_model_V1_O4.h5")
                    del modelV1

                    # load the scaler
                if not os.path.exists(f"{ann_dir}/scaler_L1_O4.pkl"):
                    scalerL1 = load_pkl_from_module('gwsnr', 'ann', 'scaler_L1_O4.pkl')
                    pickle.dump(scalerL1, open(f"{ann_dir}/scaler_L1_O4.pkl", 'wb'))
                    del scalerL1
                if not os.path.exists(f"{ann_dir}/scaler_H1_O4.pkl"):
                    scalerH1 = load_pkl_from_module('gwsnr', 'ann', 'scaler_H1_O4.pkl')
                    pickle.dump(scalerH1, open(f"{ann_dir}/scaler_H1_O4.pkl", 'wb'))
                    del scalerH1
                if not os.path.exists(f"{ann_dir}/scaler_V1_O4.pkl"):
                    scalerV1 = load_pkl_from_module('gwsnr', 'ann', 'scaler_V1_O4.pkl')
                    pickle.dump(scalerV1, open(f"{ann_dir}/scaler_V1_O4.pkl", 'wb'))
                    del scalerV1

                # redefine the path
                self.ann_path_dict = dict(
                    L1=[f"{ann_dir}/ann_model_L1_O4.h5", f"{ann_dir}/scaler_L1_O4.pkl"],
                    H1=[f"{ann_dir}/ann_model_H1_O4.h5", f"{ann_dir}/scaler_H1_O4.pkl"],
                    V1=[f"{ann_dir}/ann_model_V1_O4.h5", f"{ann_dir}/scaler_V1_O4.pkl"],
                )

            # dealing with interpolator
            self.waveform_approximant = "IMRPhenomD"
            print(
                "Please be patient while the interpolator is generated for partialscaledSNR."
            )
            path_interpolator_all = interpolator_setup()
            self.waveform_approximant = waveform_approximant

                # bool_ann = True
                # bool_ann &= self.mtot_min == 2.0
                # bool_ann &= self.mtot_max <= 184.98599853446768
                # bool_ann &= self.ratio_min == 0.1
                # bool_ann &= self.ratio_max == 1.0
                # bool_ann &= self.waveform_approximant == "IMRPhenomXPHM"
                # bool_ann &= self.sampling_frequency == 2048.0
                # bool_ann &= self.f_min == 20.0
                # bool_ann &= detector_list == ["L1", "H1", "V1"]
                # try:
                #     bool_ann &= psds_list[0].asd_file[-21:] == "aLIGO_O4_high_asd.txt"
                #     bool_ann &= psds_list[1].asd_file[-21:] == "aLIGO_O4_high_asd.txt"
                #     bool_ann &= psds_list[2].asd_file[-11:] == "AdV_asd.txt"
                # except:
                #     bool_ann = False

                # if not bool_ann:
                #     raise ValueError(
                #         "The given input parameters are not suitable for ANN method."
                #     )
                # else:
                #     print("ANN method is selected.")
                #     # dealing with interpolator
                #     self.waveform_approximant = "IMRPhenomD"
                #     print(
                #         "Please be patient while the interpolator is generated of partialscaledSNR for IMRPhenomD."
                #     )
                #     path_interpolator_all = interpolator_setup()
                #     self.waveform_approximant = "IMRPhenomXPHM"

        # change back to original
        self.psds_list = psds_list
        self.detector_tensor_list = detector_tensor_list
        self.detector_list = detector_list
        self.multiprocessing_verbose = multiprocessing_verbose
        if snr_type == "interpolation":
            self.path_interpolator = path_interpolator_all

        def print_no_interpolator(**kwargs):
            print(
                "No interpolator found. Please set snr_type=True to generate new interpolator."
            )

        if snr_type == "inner_product":
            self.snr_with_interpolation = print_no_interpolator

        # print some info
        self.print_all_params(gwsnr_verbose)

    def calculate_mtot_max(self, mtot_max, minimum_frequency):
        """
        Function to calculate maximum total mass of the binary in solar mass (use in interpolation purpose) according to the minimum frequency of the waveform.

        Parameters
        ----------
        mtot_max : `float`
            Maximum total mass of the binary in solar mass (use interpolation purpose).
        minimum_frequency : `float`
            Minimum frequency of the waveform.

        Returns
        ----------
        mtot_max : `float`
            Maximum total mass of the binary in solar mass (use interpolation purpose).
        """

        def func(x, mass_ratio=1.0):
            mass_1 = x / (1 + mass_ratio)
            mass_2 = x / (1 + mass_ratio) * mass_ratio

            return findchirp_chirptime(mass_1, mass_2, minimum_frequency) * 1.2

        # find where func is zero
        mtot_max_generated = fsolve(func, 150)[
            0
        ]  # to make sure that chirptime is not negative, TaylorF2 might need this
        if mtot_max > mtot_max_generated:
            mtot_max = mtot_max_generated

        return mtot_max

    def print_all_params(self, verbose=True):
        """
        Function to print all the parameters of the class instance

        Parameters
        ----------
        verbose : `bool`
            If True, print all the parameters of the class instance. Default is True.
        """

        if verbose:
            print("\nChosen GWSNR initialization parameters:\n")
            print("npool: ", self.npool)
            print("snr type: ", self.snr_type)
            print("waveform approximant: ", self.waveform_approximant)
            print("sampling frequency: ", self.sampling_frequency)
            print("minimum frequency (fmin): ", self.f_min)
            print("mtot=mass1+mass2")
            print("min(mtot): ", self.mtot_min)
            print(
                f"max(mtot) (with the given fmin={self.f_min}): {self.mtot_max}",
            )
            print("detectors: ", self.detector_list)
            print("psds: ", self.psds_list)
            if self.snr_type == "interpolation":
                print("min(ratio): ", self.ratio_min)
                print("max(ratio): ", self.ratio_max)
                print("mtot resolution: ", self.mtot_resolution)
                print("ratio resolution: ", self.ratio_resolution)
                print("interpolator directory: ", self.interpolator_dir)

    def snr(
        self,
        mass_1=10.0,
        mass_2=10.0,
        luminosity_distance=100.0,
        theta_jn=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=1246527224.169434,
        ra=0.0,
        dec=0.0,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        gw_param_dict=False,
        output_jsonfile=False,
    ):
        """
        Function for calling SNR calculation function depending on the value of snr_type attribute. If snr_type is 'interpolation', it calls snr_with_interpolation function. If snr_type is 'inner_product', it calls compute_bilby_snr function.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar mass. Default is 10.0.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar mass. Default is 10.0.
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle of the binary in radian. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Polarization angle of the binary in radian. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Phase of the binary in radian. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            Geocentric time of the binary in gps. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the binary in radian. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the binary in radian. Default is 0.0.
        a_1 : `numpy.ndarray` or `float`
            Primary spin of the binary. Default is 0.0.
        a_2 : `numpy.ndarray` or `float`
            Secondary spin of the binary. Default is 0.0.
        tilt_1 : `numpy.ndarray` or `float`
            Tilt of the primary spin of the binary. Default is 0.0.
        tilt_2 : `numpy.ndarray` or `float`
            Tilt of the secondary spin of the binary. Default is 0.0.
        phi_12 : `numpy.ndarray` or `float`
            Relative angle between the primary and secondary spin of the binary. Default is 0.0.
        phi_jl : `numpy.ndarray` or `float`
            Angle between the total angular momentum and the orbital angular momentum of the binary. Default is 0.0.
        gw_param_dict : `dict`
            This allows to pass all the parameters as a dictionary (dict.keys()=param_names, dict.values()=param values). Default is False.
        output_jsonfile : `str` or `bool`
            If str, the SNR dictionary will be saved as a json file with the given name. Default is False.

        Returns
        -------
        snr_dict : `dict`
            Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).

        Examples
        ----------
        >>> from gwsnr import GWSNR
        >>> snr = GWSNR(snrs_type='interpolation')
        >>> snr.snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
        """

        # if gw_param_dict is given, then use that
        if gw_param_dict is not False:
            mass_1 = gw_param_dict.get("mass_1", np.array([10.0]))
            mass_2 = gw_param_dict.get("mass_2", np.array([10.0]))
            luminosity_distance = gw_param_dict.get("luminosity_distance", np.array([100.0]))
            theta_jn = gw_param_dict.get("theta_jn", np.array([0.0]))
            psi = gw_param_dict.get("psi", np.array([0.0]))
            phase = gw_param_dict.get("phase", np.array([0.0]))
            geocent_time = gw_param_dict.get("geocent_time", np.array([1246527224.169434]))
            ra = gw_param_dict.get("ra", np.array([0.0]))
            dec = gw_param_dict.get("dec", np.array([0.0]))
            size = len(mass_1)

            # Extract spin parameters or initialize to zeros
            a_1 = gw_param_dict.get("a_1", np.zeros(size))
            a_2 = gw_param_dict.get("a_2", np.zeros(size))

            # Extract precessing waveform parameters or initialize to zeros
            tilt_1 = gw_param_dict.get("tilt_1", np.zeros(size))
            tilt_2 = gw_param_dict.get("tilt_2", np.zeros(size))
            phi_12 = gw_param_dict.get("phi_12", np.zeros(size))
            phi_jl = gw_param_dict.get("phi_jl", np.zeros(size))

        if self.snr_type == "interpolation":
            snr_dict = self.snr_with_interpolation(
                mass_1,
                mass_2,
                luminosity_distance=luminosity_distance,
                theta_jn=theta_jn,
                psi=psi,
                phase=phase,
                geocent_time=geocent_time,
                ra=ra,
                dec=dec,
                output_jsonfile=output_jsonfile,
            )

        elif self.snr_type == "inner_product":
            print("solving SNR with inner product")

            snr_dict = self.compute_bilby_snr(
                mass_1,
                mass_2,
                luminosity_distance=luminosity_distance,
                theta_jn=theta_jn,
                psi=psi,
                phase=phase,
                geocent_time=geocent_time,
                ra=ra,
                dec=dec,
                a_1=a_1,
                a_2=a_2,
                tilt_1=tilt_1,
                tilt_2=tilt_2,
                phi_12=phi_12,
                phi_jl=phi_jl,
                output_jsonfile=output_jsonfile,
            )

        elif self.snr_type == "ann":
            snr_dict = self.snr_with_ann(
                mass_1,
                mass_2,
                luminosity_distance=luminosity_distance,
                theta_jn=theta_jn,
                psi=psi,
                phase=phase,
                geocent_time=geocent_time,
                ra=ra,
                dec=dec,
                output_jsonfile=output_jsonfile,
            )

        else:
            raise ValueError("SNR function type not recognised")
        
        if self.pdet:
            pdet_dict = self.probability_of_detection(snr_dict=snr_dict, snr_th=8.0, snr_th_net=8.0, type=self.pdet)

            return pdet_dict
        else:
            return snr_dict

    def snr_with_ann(
        self,
        mass_1,
        mass_2,
        luminosity_distance=100.0,
        theta_jn=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=1246527224.169434,
        ra=0.0,
        dec=0.0,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        output_jsonfile=False,
    ):
        """
        Function to calculate SNR using bicubic interpolation method.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar mass. Default is 10.0.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar mass. Default is 10.0.
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle of the binary in radian. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Polarization angle of the binary in radian. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Phase of the binary in radian. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            Geocentric time of the binary in gps. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the binary in radian. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the binary in radian. Default is 0.0.
        output_jsonfile : `str` or `bool`
            If str, the SNR dictionary will be saved as a json file with the given name. Default is False.

        Returns
        -------
        snr_dict : `dict`
            Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).

        Examples
        ----------
        >>> from gwsnr import GWSNR
        >>> snr = GWSNR(snr_type='ann', waveform_approximant='IMRPhenomXPHM')
        >>> snr.snr_with_ann(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
        """

        # setting up the parameters
        detectors = np.array(self.detector_list)
        size = len(mass_1)
        # this allows mass_1, mass_2 to pass as float or array
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        # Broadcasting parameters to the desired size
        (
            mass_1,
            mass_2,
            luminosity_distance,
            theta_jn,
            psi,
            phase,
            geocent_time,
            ra,
            dec,
            a_1,
            a_2,
            tilt_1,
            tilt_2,
            phi_12,
            phi_jl,
        ) = np.broadcast_arrays(
            mass_1,
            mass_2,
            luminosity_distance,
            theta_jn,
            psi,
            phase,
            geocent_time,
            ra,
            dec,
            a_1,
            a_2,
            tilt_1,
            tilt_2,
            phi_12,
            phi_jl,
        )

        mtot = mass_1 + mass_2
        idx2 = np.logical_and(mtot >= self.mtot_min, mtot <= self.mtot_max)
        idx_tracker = np.nonzero(idx2)[0]
        size_ = len(idx_tracker)
        if size_ == 0:
            raise ValueError(
                "mass_1 and mass_2 must be within the range of mtot_min and mtot_max"
            )

        # output data
        params = dict(
            mass_1=mass_1,
            mass_2=mass_2,
            luminosity_distance=luminosity_distance,
            theta_jn=theta_jn,
            psi=psi,
            phase=phase,
            geocent_time=geocent_time,
            ra=ra,
            dec=dec,
            a_1=a_1,
            a_2=a_2,
            tilt_1=tilt_1,
            tilt_2=tilt_2,
            phi_12=phi_12,
            phi_jl=phi_jl,
        )
        
        # ann inputs for all detectors
        ann_input = self.output_ann(idx2, params)

        # 1. load the model 2. load feature scaler 3. predict snr
        optimal_snr = {det: np.zeros(size) for det in detectors}
        optimal_snr["optimal_snr_net"] = np.zeros(size)
        for i, det in enumerate(detectors):
            model = load_model(self.ann_path_dict[det][0])
            scaler = pickle.load(open(self.ann_path_dict[det][1], 'rb'))
            x = scaler.transform(ann_input[i])
            optimal_snr[det][idx_tracker] = model.predict(x, verbose=0).flatten()
            optimal_snr["optimal_snr_net"] += optimal_snr[det] ** 2
        optimal_snr["optimal_snr_net"] = np.sqrt(optimal_snr["optimal_snr_net"])

        # # load the model
        # modelL1 = load_h5_from_module('gwsnr', 'ann', 'ann_modelL1.h5')
        # modelH1 = load_h5_from_module('gwsnr', 'ann', 'ann_modelH1.h5')
        # modelV1 = load_h5_from_module('gwsnr', 'ann', 'ann_modelV1.h5')
        # # load the scaler
        # scalerL1 = load_pkl_from_module('gwsnr', 'ann', 'scalerL1.pkl')
        # scalerH1 = load_pkl_from_module('gwsnr', 'ann', 'scalerH1.pkl')
        # scalerV1 = load_pkl_from_module('gwsnr', 'ann', 'scalerV1.pkl')

        # # predict the output
        # # supress printing
        # snr = []
        # x = scalerL1.transform(X_L1)
        # snr.append(modelL1.predict(x,verbose = 0).flatten())
        # x = scalerH1.transform(X_H1)
        # snr.append(modelH1.predict(x, verbose = 0).flatten())
        # x = scalerV1.transform(X_V1)
        # snr.append(modelV1.predict(x, verbose = 0).flatten())
        # calculate the effective snr
        # snr_effective = np.sqrt(snr[0] ** 2 + snr[1] ** 2 + snr[2] ** 2)

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json_dict(optimal_snr, output_filename)

        return optimal_snr

    def output_ann(self, idx, params):
        """
        Function to output the input data for ANN.

        Parameters
        ----------
        idx : `numpy.ndarray`
            Index array.
        params : `dict`
            Dictionary of input parameters.

        Returns
        -------
        X_L1 : `numpy.ndarray`
            Feature scaled input data for L1 detector.
        X_H1 : `numpy.ndarray`
            Feature scaled input data for H1 detector.
        X_V1 : `numpy.ndarray`
            Feature scaled input data for V1 detector.
        """

        mass_1 = np.array(params["mass_1"][idx])
        mass_2 = np.array(params["mass_2"][idx])
        Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
        eta = mass_1 * mass_2 / (mass_1 + mass_2) ** 2.0
        A1 = Mc ** (5.0 / 6.0)

        _, _, snr_partial, d_eff = get_interpolated_snr(
            mass_1,
            mass_2,
            params["luminosity_distance"][idx],
            params["theta_jn"][idx],
            params["psi"][idx],
            params["geocent_time"][idx],
            params["ra"][idx],
            params["dec"][idx],
            np.array(self.detector_tensor_list),
            np.array(self.snr_partialsacaled_list),
            np.array(self.ratio_arr),
            np.array(self.mtot_arr),
        )

        # amp0
        amp0 = A1 / d_eff

        # get spin parameters
        a_1 = np.array(params["a_1"][idx])
        a_2 = np.array(params["a_2"][idx])
        tilt_1 = np.array(params["tilt_1"][idx])
        tilt_2 = np.array(params["tilt_2"][idx])
        # phi_12 = np.array(params["phi_12"][idx])
        # phi_jl = np.array(params["phi_jl"][idx])

        # effective spin
        chi_eff = (mass_1 * a_1 * np.cos(tilt_1) + mass_2 * a_2 * np.cos(tilt_2)) / (mass_1 + mass_2)

        # for the detectors
        ann_input = []
        for i, det in enumerate(self.detector_list):
            ann_input.append(
                np.vstack([snr_partial[i], amp0[i], eta, chi_eff]).T
            )

        # input data
        # X_L1 = np.vstack(
        #     [snr_partial[0], amp0[0], eta, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl]
        # ).T
        # X_H1 = np.vstack(
        #     [snr_partial[1], amp0[1], eta, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl]
        # ).T
        # X_V1 = np.vstack(
        #     [snr_partial[2], amp0[2], eta, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl]
        # ).T

        return (ann_input)

    def snr_with_interpolation(
        self,
        mass_1,
        mass_2,
        luminosity_distance=100.0,
        theta_jn=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=1246527224.169434,
        ra=0.0,
        dec=0.0,
        output_jsonfile=False,
    ):
        """
        Function to calculate SNR using bicubic interpolation method.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar mass. Default is 10.0.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar mass. Default is 10.0.
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle of the binary in radian. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Polarization angle of the binary in radian. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Phase of the binary in radian. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            Geocentric time of the binary in gps. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the binary in radian. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the binary in radian. Default is 0.0.
        output_jsonfile : `str` or `bool`
            If str, the SNR dictionary will be saved as a json file with the given name. Default is False.

        Returns
        -------
        snr_dict : `dict`
            Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).

        Examples
        ----------
        >>> from gwsnr import GWSNR
        >>> snr = GWSNR(snr_type='interpolation')
        >>> snr.snr_with_interpolation(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
        """

        # setting up the parameters
        detector_tensor = np.array(self.detector_tensor_list)
        detectors = np.array(self.detector_list)
        snr_partialscaled = np.array(self.snr_partialsacaled_list)
        # this allows mass_1, mass_2 to pass as float or array
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        size = len(mass_1)
        # Broadcasting parameters to the desired size
        (
            mass_1,
            mass_2,
            luminosity_distance,
            theta_jn,
            psi,
            phase,
            geocent_time,
            ra,
            dec,
        ) = np.broadcast_arrays(
            mass_1,
            mass_2,
            luminosity_distance,
            theta_jn,
            psi,
            phase,
            geocent_time,
            ra,
            dec,
        )

        mtot = mass_1 + mass_2
        idx2 = np.logical_and(mtot >= self.mtot_min, mtot <= self.mtot_max)
        idx_tracker = np.nonzero(idx2)[0]
        size_ = len(idx_tracker)
        # if size_ == 0:
        #     raise ValueError(
        #         "mass_1 and mass_2 must be within the range of mtot_min and mtot_max"
        #     )

        # Get interpolated SNR
        snr, snr_effective, _, _ = get_interpolated_snr(
            mass_1[idx2],
            mass_2[idx2],
            luminosity_distance[idx2],
            theta_jn[idx2],
            psi[idx2],
            geocent_time[idx2],
            ra[idx2],
            dec[idx2],
            detector_tensor,
            snr_partialscaled,
            self.ratio_arr,
            self.mtot_arr,
        )

        # Create optimal_snr dictionary using dictionary comprehension
        optimal_snr = {det: np.zeros(size) for det in detectors}
        optimal_snr["optimal_snr_net"] = np.zeros(size)
        for j, det in enumerate(detectors):
            optimal_snr[det][idx_tracker] = snr[j]
        optimal_snr["optimal_snr_net"][idx_tracker] = snr_effective

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json_dict(optimal_snr, output_filename)

        return optimal_snr

    def init_partialscaled(self):
        """
        Function to generate partialscaled SNR interpolation coefficients. It will save the interpolator in the pickle file path indicated by the path_interpolator attribute.
        """

        mtot_min = self.mtot_min
        detectors = self.detector_list.copy()
        detector_tensor = self.detector_tensor_list.copy()
        num_det = np.arange(len(detectors), dtype=int)
        mtot_table = self.mtot_arr
        print(f"Generating interpolator for {detectors} detectors")

        if mtot_min < 1.0:
            raise ValueError("Error: mass too low")

        # geocent_time cannot be array here
        # this geocent_time is only to get partialScaledSNR
        geocent_time_ = 1246527224.169434  # random time from O3
        theta_jn_, ra_, dec_, psi_, phase_ = np.zeros(5)
        luminosity_distance_ = 100.0

        # Vectorized computation for effective luminosity distance
        Fp = np.array(
            [
                antenna_response(ra_, dec_, geocent_time_, psi_, tensor, "plus")
                for tensor in detector_tensor
            ]
        )
        Fc = np.array(
            [
                antenna_response(ra_, dec_, geocent_time_, psi_, tensor, "cross")
                for tensor in detector_tensor
            ]
        )
        dl_eff = luminosity_distance_ / np.sqrt(
            Fp**2 * ((1 + np.cos(theta_jn_) ** 2) / 2) ** 2
            + Fc**2 * np.cos(theta_jn_) ** 2
        )

        ratio = self.ratio_arr.copy()
        snr_partial_ = []
        # interpolation along mtot for each mass_ratio
        for q in tqdm(
            ratio,
            desc="interpolation for each mass_ratios",
            total=len(ratio),
            ncols=100,
        ):
            mass_1_ = mtot_table / (1 + q)
            mass_2_ = mass_1_ * q
            # calling bilby_snr
            optimal_snr_unscaled = self.compute_bilby_snr(
                mass_1=mass_1_,
                mass_2=mass_2_,
                luminosity_distance=luminosity_distance_,
                theta_jn=theta_jn_,
                psi=psi_,
                phase=phase_,
                geocent_time=geocent_time_,
                ra=ra_,
                dec=dec_,
            )
            # for partialscaledSNR
            mchirp = ((mass_1_ * mass_2_) ** (3 / 5)) / ((mtot_table) ** (1 / 5))
            a2 = mchirp ** (5.0 / 6.0)
            # filling in interpolation table for different detectors
            snr_partial_buffer = []
            for j in num_det:
                snr_partial_buffer.append(
                    CubicSpline(
                        mtot_table,
                        (dl_eff[j] / a2) * optimal_snr_unscaled[detectors[j]],
                    ).c
                )
            snr_partial_.append(np.array(snr_partial_buffer))
        snr_partial_ = np.array(snr_partial_)

        # save the interpolators for each detectors
        for j in num_det:
            save_json(snr_partial_[:, j], self.path_interpolator[j])

    def compute_bilby_snr(
        self,
        mass_1=10,
        mass_2=10,
        luminosity_distance=100.0,
        theta_jn=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=1246527224.169434,
        ra=0.0,
        dec=0.0,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        gw_param_dict=False,
        output_jsonfile=False,
    ):
        """
        SNR calculated using inner product method. This is similar to the SNR calculation method used in bilby.

        Parameters
        ----------
        mass_1 : float
            The mass of the heavier object in the binary in solar masses.
        mass_2 : float
            The mass of the lighter object in the binary in solar masses.
        luminosity_distance : float
            The luminosity distance to the binary in megaparsecs.
        theta_jn : float, optional
            The angle between the total angular momentum and the line of sight.
            Default is 0.
        psi : float, optional
            The gravitational wave polarisation angle.
            Default is 0.
        phase : float, optional
            The gravitational wave phase at coalescence.
            Default is 0.
        geocent_time : float, optional
            The GPS time of coalescence.
            Default is 1249852157.0.
        ra : float, optional
            The right ascension of the source.
            Default is 0.
        dec : float, optional
            The declination of the source.
            Default is 0.
        a_1 : float, optional
            The spin magnitude of the heavier object in the binary.
            Default is 0.
        tilt_1 : float, optional
            The tilt angle of the heavier object in the binary.
            Default is 0.
        phi_12 : float, optional
            The azimuthal angle between the two spins.
            Default is 0.
        a_2 : float, optional
            The spin magnitude of the lighter object in the binary.
            Default is 0.
        tilt_2 : float, optional
            The tilt angle of the lighter object in the binary.
            Default is 0.
        phi_jl : float, optional
            The azimuthal angle between the total angular momentum and the orbital angular momentum.
            Default is 0.
        verbose : bool, optional
            If true, print the SNR.
            Default is True.
        jsonFile : bool, optional
            If true, save the SNR parameters and values in a json file.
            Default is False.

        Returns
        ----------
        snr_dict : `dict`
            Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).

        Examples
        ----------
        >>> from gwsnr import GWSNR
        >>> snr = GWSNR(snrs_type='inner_product')
        >>> snr.compute_bilby_snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
        """

        # if gw_param_dict is given, then use that
        if gw_param_dict is not False:
            mass_1 = gw_param_dict["mass_1"]
            mass_2 = gw_param_dict["mass_2"]
            luminosity_distance = gw_param_dict["luminosity_distance"]
            theta_jn = gw_param_dict["theta_jn"]
            psi = gw_param_dict["psi"]
            phase = gw_param_dict["phase"]
            geocent_time = gw_param_dict["geocent_time"]
            ra = gw_param_dict["ra"]
            dec = gw_param_dict["dec"]
            # a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl exist in the dictionary
            # if exists, then use that, else pass
            if "a_1" and "a2" in gw_param_dict:
                a_1 = gw_param_dict["a_1"]
                a_2 = gw_param_dict["a_2"]
            if "tilt_1" and "tilt_2" and "phi_12" and "phi_jl" in gw_param_dict:
                tilt_1 = gw_param_dict["tilt_1"]
                tilt_2 = gw_param_dict["tilt_2"]
                phi_12 = gw_param_dict["phi_12"]
                phi_jl = gw_param_dict["phi_jl"]

        npool = self.npool
        sampling_frequency = self.sampling_frequency
        detectors = self.detector_list.copy()
        detector_tensor = np.array(self.detector_tensor_list.copy())
        approximant = self.waveform_approximant
        f_min = self.f_min
        num_det = np.arange(len(detectors), dtype=int)

        # get the psds for the required detectors
        psd_dict = {detectors[i]: self.psds_list[i] for i in num_det}

        # reshape(-1) is so that either a float value is given or the input is an numpy array
        # make sure all parameters are of same length
        num = len(mass_1)
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        (
            mass_1,
            mass_2,
            luminosity_distance,
            theta_jn,
            psi,
            phase,
            geocent_time,
            ra,
            dec,
            a_1,
            a_2,
            tilt_1,
            tilt_2,
            phi_12,
            phi_jl,
        ) = np.broadcast_arrays(
            mass_1,
            mass_2,
            luminosity_distance,
            theta_jn,
            psi,
            phase,
            geocent_time,
            ra,
            dec,
            a_1,
            a_2,
            tilt_1,
            tilt_2,
            phi_12,
            phi_jl,
        )

        #############################################
        # setting up parameters for multiprocessing #
        #############################################
        mtot = mass_1 + mass_2
        idx = (mtot >= self.mtot_min) & (mtot <= self.mtot_max)
        size1 = np.sum(idx)
        iterations = np.arange(size1)  # to keep track of index

        dectector_arr = np.array(detectors) * np.ones(
            (size1, len(detectors)), dtype=object
        )
        psds_dict_list = np.array([np.full(size1, psd_dict, dtype=object)]).T
        # IMPORTANT: time duration calculation for each of the mass combination
        safety = 1.2
        approx_duration = safety * findchirp_chirptime(mass_1[idx], mass_2[idx], f_min)
        duration = np.ceil(approx_duration + 2.0)

        input_arguments = np.array(
            [
                mass_1[idx],
                mass_2[idx],
                luminosity_distance[idx],
                theta_jn[idx],
                psi[idx],
                phase[idx],
                ra[idx],
                dec[idx],
                geocent_time[idx],
                a_1[idx],
                a_2[idx],
                tilt_1[idx],
                tilt_2[idx],
                phi_12[idx],
                phi_jl[idx],
                np.full(size1, approximant),
                np.full(size1, f_min),
                duration,
                np.full(size1, sampling_frequency),
                iterations,
            ],
            dtype=object,
        ).T

        input_arguments = np.concatenate(
            (input_arguments, psds_dict_list, dectector_arr), axis=1
        )

        # np.shape(hp_inner_hp) = (len(num_det), size1)
        hp_inner_hp = np.zeros((len(num_det), size1), dtype=np.complex128)
        hc_inner_hc = np.zeros((len(num_det), size1), dtype=np.complex128)
        with Pool(processes=npool) as pool:
            # call the same function with different data in parallel
            # imap->retain order in the list, while map->doesn't
            if self.multiprocessing_verbose:
                for result in tqdm(
                    pool.imap_unordered(noise_weighted_inner_prod, input_arguments),
                    total=len(input_arguments),
                    ncols=100,
                ):
                    # but, np.shape(hp_inner_hp_i) = (size1, len(num_det))
                    hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                    hp_inner_hp[:, iter_i] = hp_inner_hp_i
                    hc_inner_hc[:, iter_i] = hc_inner_hc_i
            else:
                # with map, without tqdm
                for result in pool.map(noise_weighted_inner_prod, input_arguments):
                    hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                    hp_inner_hp[:, iter_i] = hp_inner_hp_i
                    hc_inner_hc[:, iter_i] = hc_inner_hc_i

        # get polarization tensor
        # np.shape(Fp) = (size1, len(num_det))
        Fp, Fc = antenna_response_array(
            ra[idx], dec[idx], geocent_time[idx], psi[idx], detector_tensor
        )
        snrs_sq = abs((Fp**2) * hp_inner_hp + (Fc**2) * hc_inner_hc)
        snr = np.sqrt(snrs_sq)
        snr_effective = np.sqrt(np.sum(snrs_sq, axis=0))

        # organizing the snr dictionary
        optimal_snr = dict()
        for j, det in enumerate(detectors):
            snr_buffer = np.zeros(num)
            snr_buffer[idx] = snr[j]
            optimal_snr[det] = snr_buffer
        snr_buffer = np.zeros(num)
        snr_buffer[idx] = snr_effective
        optimal_snr["optimal_snr_net"] = snr_buffer

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json_dict(optimal_snr, output_filename)

        return optimal_snr

    def probability_of_detection(self, snr_dict, snr_th=None, snr_th_net=None, type="matched_filter"):
        """
        Probaility of detection of GW for the given sensitivity of the detectors

        Parameters
        ----------
        snr_dict : `dict`
            Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).
        rho_th : `float`
            Threshold SNR for detection. Default is 8.0.
        rho_net_th : `float`
            Threshold net SNR for detection. Default is 8.0.
        type : `str`
            Type of SNR calculation. Default is 'matched_filter'. Other option is 'bool'.

        Returns
        ----------
        pdet_dict : `dict`
            Dictionary of probability of detection for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=pdet_arrays).
        """

        if snr_th:
            snr_th = snr_th
        else:
            snr_th = self.snr_th

        if snr_th_net:
            snr_th_net = snr_th_net
        else:
            snr_th_net = self.snr_th_net

        detectors = np.array(self.detector_list)
        pdet_dict = {}
        for det in detectors:
            if type == "matched_filter":
                pdet_dict[det] = np.array(1 - norm.cdf(snr_th - snr_dict[det]), dtype=int)
            else:
                pdet_dict[det] = np.array(snr_th < snr_dict[det], dtype=int)

        if type == "matched_filter":
            pdet_dict["pdet_net"] = np.array(1 - norm.cdf(snr_th_net - snr_dict["optimal_snr_net"]), dtype=int)
        else:
            pdet_dict["pdet_net"] = np.array(snr_th_net < snr_dict["optimal_snr_net"], dtype=int)

        return pdet_dict

    def detector_horizon(self, mass_1=1.4, mass_2=1.4, snr_th=None, snr_th_net=None):
        """
        Function for finding detector horizon distance for BNS (m1=m2=1.4)

        Parameters
        ----------
        mass_1 : `float`
            Primary mass of the binary in solar mass. Default is 1.4.
        mass_2 : `float`
            Secondary mass of the binary in solar mass. Default is 1.4.
        snr_th : `float`
            SNR threshold for detection. Default is 8.0.

        Returns
        ----------
        horizon : `dict`
            Dictionary of horizon distance for each detector (dict.keys()=detector_names, dict.values()=horizon_distance).
        """

        if snr_th:
            snr_th = snr_th
        else:
            snr_th = self.snr_th

        if snr_th_net:
            snr_th_net = snr_th_net
        else:
            snr_th_net = self.snr_th_net

        detectors = np.array(self.detector_list.copy())
        detector_tensor = np.array(self.detector_tensor_list.copy())
        geocent_time_ = 1246527224.169434  # random time from O3
        theta_jn_, ra_, dec_, psi_, phase_ = 0.0, 0.0, 0.0, 0.0, 0.0
        luminosity_distance_ = 100.0

        # calling bilby_snr
        optimal_snr_unscaled = self.compute_bilby_snr(
            mass_1=mass_1,
            mass_2=mass_2,
            luminosity_distance=luminosity_distance_,
            theta_jn=theta_jn_,
            psi=psi_,
            phase=phase_,
            ra=ra_,
            dec=dec_,
        )

        # Vectorized computation for effective luminosity distance
        Fp = np.array(
            [
                antenna_response(ra_, dec_, geocent_time_, psi_, tensor, "plus")
                for tensor in detector_tensor
            ]
        )
        Fc = np.array(
            [
                antenna_response(ra_, dec_, geocent_time_, psi_, tensor, "cross")
                for tensor in detector_tensor
            ]
        )
        dl_eff = luminosity_distance_ / np.sqrt(
            Fp**2 * ((1 + np.cos(theta_jn_) ** 2) / 2) ** 2
            + Fc**2 * np.cos(theta_jn_) ** 2
        )

        # Horizon calculation
        horizon = {
            det: (dl_eff[j] / snr_th) * optimal_snr_unscaled[det]
            for j, det in enumerate(detectors)
        }
        horizon["net"] = (dl_eff / snr_th_net) * optimal_snr_unscaled["optimal_snr_net"]

        return horizon