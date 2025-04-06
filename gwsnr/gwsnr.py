# -*- coding: utf-8 -*-
"""
This module contains functions for calculating the SNR of a CBC signal. It has two methods: interpolation (bicubic) and inner product (multiprocessing and jax.jit+jax.vmap). Interpolation method is much faster than inner product method. Interpolation method is tested for IMRPhenomD and TaylorF2 waveform approximants for the spinless scenario.
"""

import os
import pickle
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# warning suppression lal
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
lal.swig_redirect_standard_output_error(False)

from .ripple_class import RippleInnerProduct

from .utils import (
    dealing_with_psds,
    interpolator_check,
    load_json,
    load_pickle,
    save_pickle,
    save_json,
    load_ann_h5_from_module,
    load_ann_h5,
    load_pickle_from_module,
    load_json_from_module,
    get_gw_parameters,
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
        options: 'interpolation', 'inner_product', 'inner_product_jax', 'pdet', 'ann'
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
    |                                     | (python multiprocessing)         |
    +-------------------------------------+----------------------------------+
    |:meth:`~compute_ripple_snr`          | Calculates SNR using             |
    |                                     | inner product method.            |
    |                                     | (jax.jit+jax.vmap)               |
    +-------------------------------------+----------------------------------+
    |:meth:`~bns_horizon`                 | Calculates BNS horizon           |
    |                                     | distance.                        |
    +-------------------------------------+----------------------------------+
    |:meth:`~print_all_params`            | Prints all the parameters of     |
    |                                     | the class instance.              |
    +-------------------------------------+----------------------------------+
    |:meth:`~init_partialscaled`             | Generates partialscaled SNR   |
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
    ):

        print("\nInitializing GWSNR class...\n")
        # setting instance attributes
        self.npool = npool
        self.pdet = pdet
        self.snr_th = snr_th
        self.snr_th_net = snr_th_net
        self.duration_max = duration_max
        self.duration_min = duration_min
        self.snr_type = snr_type
        # change multiprocessing start method from fork to spawn if snr_type is inner_product_jax
        if self.snr_type == "inner_product_jax":
            import multiprocessing as mp
            import os

            def set_multiprocessing_start_method():
                start_method = 'spawn'  # Use 'spawn' for both POSIX and non-POSIX systems
                try:
                    mp.set_start_method(start_method, force=True)
                except RuntimeError:
                    # The start method can only be set once and must be set before any processes start
                    pass
            set_multiprocessing_start_method()

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
        self.frequency_domain_source_model = frequency_domain_source_model
        self.f_min = minimum_frequency
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
        if waveform_approximant=="IMRPhenomXPHM" and duration_max is None:
            print("Intel processor has trouble allocating memory when the data is huge. So, by default for IMRPhenomXPHM, duration_max = 64.0. Otherwise, set to some max value like duration_max = 600.0 (10 mins)")
            self.duration_max = 64.0
            self.durarion_min = 4.0


        # now generate interpolator, if not exists
        if snr_type == "interpolation":
            # dealing with interpolator
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

        # inner product method doesn't need interpolator generation
        elif snr_type == "inner_product":
            pass
        
        # need to initialize RippleInnerProduct class
        elif snr_type == "inner_product_jax":
            ripple_class = RippleInnerProduct(
                waveform_name=waveform_approximant, 
                minimum_frequency=minimum_frequency, 
                sampling_frequency=sampling_frequency, 
                reference_frequency=minimum_frequency
                )

            self.noise_weighted_inner_product_jax = ripple_class.noise_weighted_inner_product_jax

        # ANN method still needs the partialscaledSNR interpolator.
        elif snr_type == "ann":
            self.model_dict, self.scaler_dict, self.error_adjustment, self.ann_catalogue = self.ann_initilization(ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant, snr_th)
            # dealing with interpolator
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

        else:
            raise ValueError("SNR function type not recognised. Please choose from 'interpolation', 'inner_product', 'inner_product_jax', 'ann'.")

        # change back to original
        self.psds_list = psds_list
        self.detector_tensor_list = detector_tensor_list
        self.detector_list = detector_list
        self.multiprocessing_verbose = multiprocessing_verbose

        if (snr_type == "inner_product") or (snr_type == "inner_product_jax"):
            self.snr_with_interpolation = self._print_no_interpolator

        # print some info
        self.print_all_params(gwsnr_verbose)

    # dealing with interpolator
    def interpolator_setup(self, interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list):
        """
        Function to generate the partialscaled SNR interpolator and return its pickle file paths.

        Parameters
        ----------
        interpolator_dir : `str`
            Path to store the interpolator pickle file.
        create_new_interpolator : `bool`
            If set True, new interpolator will be generated or replace the existing one.
        psds_list : `list`
            List of psds for different detectors.
        detector_tensor_list : `list`
            List of detector tensor.
        detector_list : `list`
            List of detectors.


        Returns
        ----------
        path_interpolator_all : `list`
            List of partialscaled SNR interpolator pickle file paths.
        """

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
            load_pickle(path) for path in path_interpolator_all
        ]

        return path_interpolator_all

    def ann_initilization(self, ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant, snr_th):
        """
        Function to initialize ANN model and scaler for the given detector list. It also generates the partialscaledSNR interpolator for the required waveform approximant.

        """

        # check the content ann_path_dict.json in gwsnr/ann module directory
        # e.g. ann_path_dict = dict(L1=dict(model_path='path_to_model', scaler_path='path_to_scaler', sampling_frequency=2048.0, minimum_frequency=20.0, waveform_approximant='IMRPhenomXPHM', snr_th=8.0))
        # there will be existing ANN model and scaler for default parameters
        ann_path_dict_default = load_json_from_module('gwsnr', 'ann', 'ann_path_dict.json')
        if ann_path_dict is None:
            ann_path_dict = ann_path_dict_default
        else:
            if isinstance(ann_path_dict, str):
                ann_path_dict = load_json(ann_path_dict)
            elif isinstance(ann_path_dict, dict):
                pass
            else:
                raise ValueError("ann_path_dict should be a dictionary or a path to the json file.")
            # if 'L1' key already exist in the dict, you can still give new dict value for 'L1' when initializing GWSNR.
            # if no new dict value is given for 'L1', it will take the default value from ann_path_dict_default.
            ann_path_dict_default.update(ann_path_dict)
            ann_path_dict = ann_path_dict_default
        del ann_path_dict_default

        model_dict = {}
        scaler_dict = {}
        error_adjustment = {}
        # loop through the detectors
        for detector in detector_list:
            if detector not in ann_path_dict.keys():
                # check if the model and scaler is available
                raise ValueError(f"ANN model and scaler for {detector} is not available. Please provide the path to the model and scaler. Refer to the 'gwsnr' documentation for more information on how to add new ANN model.")
            else:
                # check of model parameters
                check = True
                check &= (sampling_frequency == ann_path_dict[detector]['sampling_frequency'])
                check &= (minimum_frequency == ann_path_dict[detector]['minimum_frequency'])
                check &= (waveform_approximant == ann_path_dict[detector]['waveform_approximant'])
                check &= (snr_th == ann_path_dict[detector]['snr_th'])
                # check for the model and scaler keys exit or not
                check &= ('model_path' in ann_path_dict[detector].keys())
                check &= ('scaler_path' in ann_path_dict[detector].keys())

                if not check:
                    raise ValueError(f"ANN model parameters for {detector} is not suitable for the given gwsnr parameters. Existing parameters are: {ann_path_dict[detector]}")

            # get ann model
            if not os.path.exists(ann_path_dict[detector]['model_path']):
                # load the model from gwsnr/ann directory
                model_dict[detector] = load_ann_h5_from_module('gwsnr', 'ann', ann_path_dict[detector]['model_path'])
                print(f"ANN model for {detector} is loaded from gwsnr/ann directory.")
            else:
                # load the model from the given path
                model_dict[detector] = load_ann_h5(ann_path_dict[detector]['model_path'])
                print(f"ANN model for {detector} is loaded from {ann_path_dict[detector]['model_path']}.")

            # get ann scaler
            if not os.path.exists(ann_path_dict[detector]['scaler_path']):
                # load the scaler from gwsnr/ann directory
                scaler_dict[detector] = load_pickle_from_module('gwsnr', 'ann', ann_path_dict[detector]['scaler_path'])
                print(f"ANN scaler for {detector} is loaded from gwsnr/ann directory.")
            else:
                # load the scaler from the given path
                scaler_dict[detector] = load_pickle(ann_path_dict[detector]['scaler_path'])
                print(f"ANN scaler for {detector} is loaded from {ann_path_dict[detector]['scaler_path']}.")

            # get error_adjustment
            if not os.path.exists(ann_path_dict[detector]['error_adjustment_path']):
                # load the error_adjustment from gwsnr/ann directory
                error_adjustment[detector] = load_json_from_module('gwsnr', 'ann', ann_path_dict[detector]['error_adjustment_path'])
                print(f"ANN error_adjustment for {detector} is loaded from gwsnr/ann directory.")
            else:
                # load the error_adjustment from the given path
                error_adjustment[detector] = load_json(ann_path_dict[detector]['error_adjustment_path'])
                print(f"ANN error_adjustment for {detector} is loaded from {ann_path_dict[detector]['error_adjustment_path']}.")

        return model_dict, scaler_dict, error_adjustment, ann_path_dict

    def _print_no_interpolator(self, **kwargs):
        """
        Helper function to print error message when no interpolator is found.

        Parameters
        ----------
        kwargs : `dict`
            Dictionary of parameters.
        """

        raise ValueError(
            'No interpolator found. Please set snr_type="interpolation" to generate new interpolator.'
        )

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

            return findchirp_chirptime(mass_1, mass_2, minimum_frequency) * 1.1

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
        mass_1=np.array([10.0,]),
        mass_2=np.array([10.0,]),
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
        lambda_1=0.0,
        lambda_2=0.0,
        eccentricity=0.0,
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
                gw_param_dict=gw_param_dict,
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
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                eccentricity=eccentricity,
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )

        elif self.snr_type == "inner_product_jax":
            print("solving SNR with inner product JAX")

            snr_dict = self.compute_ripple_snr(
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
                gw_param_dict=gw_param_dict,
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
                a_1=a_1,
                a_2=a_2,
                tilt_1=tilt_1,
                tilt_2=tilt_2,
                phi_12=phi_12,
                phi_jl=phi_jl,
                gw_param_dict=gw_param_dict,
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
        gw_param_dict=False,
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

        if gw_param_dict is not False:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, _, _, _ = get_gw_parameters(gw_param_dict)
        else:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, _, _, _ = get_gw_parameters(dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, ra=ra, dec=dec, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl))

        # setting up the parameters
        model = self.model_dict
        scaler = self.scaler_dict
        detectors = np.array(self.detector_list)
        size = len(mass_1)
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
            x = scaler[det].transform(ann_input[i])
            optimal_snr_ = model[det].predict(x, verbose=0).flatten()
            optimal_snr[det][idx_tracker] = optimal_snr_ - (self.error_adjustment[det]['slope']*optimal_snr_ + self.error_adjustment[det]['intercept'])
            optimal_snr["optimal_snr_net"] += optimal_snr[det] ** 2
        optimal_snr["optimal_snr_net"] = np.sqrt(optimal_snr["optimal_snr_net"])

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json(output_filename, optimal_snr)

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

        # inclination angle
        theta_jn = np.array(params["theta_jn"][idx])

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
        for i in range(len(self.detector_list)):
            ann_input.append(
                np.vstack([snr_partial[i], amp0[i], eta, chi_eff, theta_jn]).T
            )

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
        gw_param_dict=False,
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

        # getting the parameters from the dictionary
        if gw_param_dict is not False:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, _, _, _, _, _, _, _, _, _ = get_gw_parameters(gw_param_dict)
        else:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, _, _, _, _, _, _, _, _, _ = get_gw_parameters(dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, ra=ra, dec=dec))

        # setting up the parameters
        detector_tensor = np.array(self.detector_tensor_list)
        detectors = np.array(self.detector_list)
        snr_partialscaled = np.array(self.snr_partialsacaled_list)

        size = len(mass_1)
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
            save_json(output_filename, optimal_snr)

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
            Mchirp = ((mass_1_ * mass_2_) ** (3 / 5)) / ((mtot_table) ** (1 / 5))
            a2 = Mchirp ** (5.0 / 6.0)
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
            save_pickle(self.path_interpolator[j], snr_partial_[:, j])

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
        lambda_1=0.0,
        lambda_2=0.0,
        eccentricity=0.0,
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
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, eccentricity  = get_gw_parameters(gw_param_dict)
        else:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, eccentricity = get_gw_parameters(dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, ra=ra, dec=dec, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl, lambda_1=lambda_1, lambda_2=lambda_2, eccentricity=eccentricity))

        npool = self.npool
        sampling_frequency = self.sampling_frequency
        detectors = self.detector_list.copy()
        detector_tensor = np.array(self.detector_tensor_list.copy())
        approximant = self.waveform_approximant
        f_min = self.f_min
        num_det = np.arange(len(detectors), dtype=int)

        # get the psds for the required detectors
        psd_dict = {detectors[i]: self.psds_list[i] for i in num_det}
        num = len(mass_1)

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
        frequency_domain_source_model = np.array([np.full(size1, self.frequency_domain_source_model)]).T
        psds_dict_list = np.array([np.full(size1, psd_dict, dtype=object)]).T
        # IMPORTANT: time duration calculation for each of the mass combination
        safety = 1.1
        approx_duration = safety * findchirp_chirptime(mass_1[idx], mass_2[idx], f_min)
        duration = np.ceil(approx_duration + 2.0)
        if self.duration_max:
            duration[duration > self.duration_max] = self.duration_max  # IMRPheonomXPHM has maximum duration of 371s
        if self.duration_min:
            duration[duration < self.duration_min] = self.duration_min


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
                lambda_1[idx],
                lambda_2[idx],
                eccentricity[idx],
                np.full(size1, approximant),
                np.full(size1, f_min),
                duration,
                np.full(size1, sampling_frequency),
                iterations,
            ],
            dtype=object,
        ).T

        input_arguments = np.concatenate(
            (input_arguments, psds_dict_list, frequency_domain_source_model, dectector_arr), axis=1
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
            save_json(output_filename, optimal_snr)

        return optimal_snr

    def compute_ripple_snr(
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
        SNR calculated using inner product method with ripple generated waveform.

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
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, _, _, _ = get_gw_parameters(gw_param_dict)
        else:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, _, _, _ = get_gw_parameters(dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, ra=ra, dec=dec, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl))

        npool = self.npool
        detectors = self.detector_list.copy()
        detector_tensor = np.array(self.detector_tensor_list.copy())
        num_det = np.arange(len(detectors), dtype=int)
        # get the psds for the required detectors
        psd_list = self.psds_list.copy()
        num = len(mass_1)

        #############################################
        # setting up parameters for multiprocessing #
        #############################################
        mtot = mass_1 + mass_2
        idx = (mtot >= self.mtot_min) & (mtot <= self.mtot_max)
        # size1 = np.sum(idx)
        # iterations = np.arange(size1)  # to keep track of index

        input_dict = dict(
            mass_1=mass_1[idx],
            mass_2=mass_2[idx],
            luminosity_distance=luminosity_distance[idx],
            theta_jn=theta_jn[idx],
            psi=psi[idx],
            phase=phase[idx],
            geocent_time=geocent_time[idx],
            ra=ra[idx],
            dec=dec[idx],
            a_1=a_1[idx],
            a_2=a_2[idx],
            tilt_1=tilt_1[idx],
            tilt_2=tilt_2[idx],
            phi_12=phi_12[idx],
            phi_jl=phi_jl[idx],
        )

        hp_inner_hp, hc_inner_hc = self.noise_weighted_inner_product_jax(
            gw_param_dict=input_dict, 
            psd_list=psd_list,
            detector_list=detectors, 
            duration_min=self.duration_min,
            duration_max=self.duration_max,
            npool=npool,
            multiprocessing_verbose=self.multiprocessing_verbose
        )

        # gw_param_dict, psd_object_list, detector_list, duration=None, duration_min=2, duration_max=128

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
            output_filename = (output_jsonfile if isinstance(output_jsonfile, str) else "snr.json")
            save_json(output_filename, optimal_snr)

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
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar mass. Default is 1.4.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar mass. Default is 1.4.
        snr_th : `float`
            SNR threshold for detection. Default is 8.0.

        Returns
        ----------
        horizon : `dict`
            Dictionary of horizon distance for each detector in Mpc (dict.keys()=detector_names, dict.values()=horizon_distance).
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

        dl_eff = np.sqrt(np.sum(dl_eff**2))
        horizon["net"] = (dl_eff / snr_th_net) * optimal_snr_unscaled["optimal_snr_net"]
        #print('dl_eff', dl_eff)
        #print('optimal_snr_unscaled', optimal_snr_unscaled["optimal_snr_net"])

        return horizon
