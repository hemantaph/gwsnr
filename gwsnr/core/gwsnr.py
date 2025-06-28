# -*- coding: utf-8 -*-
"""
Gravitational-wave signal-to-noise ratio calculation.

This module provides efficient calculation of signal-to-noise ratio (SNR) 
for gravitational-wave signals from compact binary coalescences (CBCs).

The module implements multiple computational backends optimized for different
use cases: interpolation-based methods for fast evaluation, noise-weighted
inner products for high accuracy, JAX-accelerated computation for vectorized
operations, and artificial neural networks for rapid probability estimation.

Methods
-------
The package supports five main computational approaches:

Interpolation Method (Partial-Scaled SNR)
    Fast bicubic interpolation of precomputed partial-scaled SNRs. Efficient
    for aligned-spin or non-spinning systems. Uses grids of intrinsic 
    parameters decoupled from extrinsic parameters. Supports waveform
    approximants: IMRPhenomD, TaylorF2, IMRPhenomXPHM.

Noise-Weighted Inner Product Method  
    Standard matched-filtering SNR calculation using inner-product integral
    between waveform and noise PSD. Supports multiprocessing and waveform
    generation from lalsimulation and ripple. Compatible with arbitrary
    frequency-domain models including precession and higher harmonics.

JAX-based Inner Product
    Hardware-accelerated computation using ripple waveform generator with
    JAX jit compilation and vmap vectorization for batched evaluation.

Artificial Neural Network (ANN) Estimation
    Rapid probability of detection estimation for spin-precessing systems
    using trained neural network models. Uses partial-scaled SNR as summary
    statistic to reduce dimensionality. Supports user-supplied models.

Hybrid SNR Recalculation
    Combines fast interpolation with accurate inner-product recalculation
    for systems near detection threshold.

Features
--------
- Customizable detector configurations and power spectral densities
- Bilby interferometer and PSD interface compatibility  
- Large-scale population synthesis optimization
- Integration with gravitational-wave lensing tools
- Automated interpolator generation and caching
- Extensible neural network model framework

Examples
--------
Basic SNR calculation using interpolation:

>>> from gwsnr import GWSNR
>>> snr_calc = GWSNR(snr_type='interpolation', 
...                  waveform_approximant='IMRPhenomD')
>>> result = snr_calc.snr(mass_1=30, mass_2=30, 
...                       luminosity_distance=100, 
...                       theta_jn=0.0, ra=0.0, dec=0.0)

Custom detector configuration:

>>> import bilby
>>> ifo = bilby.gw.detector.interferometer.Interferometer(
...     name='LIO',
...     power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
...         asd_file='custom_psd.txt'),
...     minimum_frequency=10.0,
...     maximum_frequency=2048.0,
...     length=4,
...     latitude=19.613, longitude=77.031, elevation=450.0,
...     xarm_azimuth=117.6, yarm_azimuth=207.6)
>>> snr_calc = GWSNR(psds={'LIO': 'custom_psd.txt'}, ifos=[ifo])

Notes
-----
The interpolation methods are particularly efficient for population studies
involving thousands to millions of systems. For individual high-precision
calculations, the inner product methods provide the most accurate results.

Neural network estimation is experimental and currently optimized for
specific detector configurations and parameter ranges.

References
----------
.. [1] Phurailatpam & Hannuksela (2025), "gwsnr: A Python package for 
       efficient signal-to-noise calculation of gravitational-waves", 
       JOSS Publications.
.. [2] Allen et al. (2012), "FINDCHIRP: An algorithm for detection of 
       gravitational waves from inspiraling compact binaries", 
       Phys. Rev. D 85, 122006.
.. [3] Edwards et al. (2024), "Differentiable and hardware-accelerated 
       waveforms for gravitational wave data analysis", Phys. Rev. D.
.. [4] Ashton et al. (2019), "Bilby: A user-friendly Bayesian inference 
       library for gravitational-wave astronomy", Astrophys. J. Suppl. 241, 27.
"""


import shutil
import os
from importlib.resources import path
import pathlib

import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from scipy.stats import norm

# warning suppression lal
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
lal.swig_redirect_standard_output_error(False)

from ..utils import (
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
)  # from gwsnr/utils/utils.py
from ..utils import noise_weighted_inner_prod  # from gwsnr/utils/multiprocessing_routine.py

from ..numba import (
    findchirp_chirptime,
    antenna_response_plus,
    antenna_response_cross,
    antenna_response_array,
)

# defining constants
C = 299792458.0
G = 6.67408e-11
Pi = np.pi
MTSUN_SI = 4.925491025543576e-06

class GWSNR:
    """
    Class to calculate SNR of a CBC signal with either interpolation or inner product method. Interpolation method is much faster than inner product method. Interpolation method is tested for IMRPhenomD, TaylorF2, and IMRPhenomXPHM waveform approximants for both spinless and aligned-spin scenarios.

    Parameters
    ----------
    npool : `int`
        Number of processors to use for parallel processing.
        Default is 4.
    mtot_min : `float`
        Minimum total mass of the binary in solar mass (use interpolation purpose). Default is 2*4.98-2 (4.98 Mo is the minimum component mass of BBH systems in GWTC-3).
    mtot_max : `float`
        Maximum total mass of the binary in solar mass (use interpolation purpose). Default is 2*112.5+2 (112.5 Mo is the maximum component mass of BBH systems in GWTC-3).
        This is automatically adjusted based on minimum_frequency if mtot_cut=True.
    ratio_min : `float`
        Minimum mass ratio of the binary (use interpolation purpose). Default is 0.1.
    ratio_max : `float`
        Maximum mass ratio of the binary (use interpolation purpose). Default is 1.0.
    spin_max : `float`
        Maximum spin magnitude for aligned-spin interpolation methods. Default is 0.9.
    mtot_resolution : `int`
        Number of points in the total mass array (use interpolation purpose). Default is 200.
    ratio_resolution : `int`
        Number of points in the mass ratio array (use interpolation purpose). Default is 50.
    spin_resolution : `int`
        Number of points in the spin arrays for aligned-spin interpolation methods. Default is 20.
    sampling_frequency : `float`
        Sampling frequency of the detector. Default is 2048.0.
    waveform_approximant : `str`
        Waveform approximant to use. Default is 'IMRPhenomD'.
    frequency_domain_source_model : `str`
        Source model for frequency domain waveform generation. Default is 'lal_binary_black_hole'.
    minimum_frequency : `float`
        Minimum frequency of the waveform. Default is 20.0.
    duration_max : `float` or `None`
        Maximum duration for waveform generation. Default is None. Automatically set to 64.0 for IMRPhenomXPHM on Intel processors.
    duration_min : `float` or `None`
        Minimum duration for waveform generation. Default is None.
    snr_type : `str`
        Type of SNR calculation. Default is 'interpolation'.
        options: 'interpolation', 'interpolation_no_spins', 'interpolation_no_spins_jax', 'interpolation_aligned_spins', 'interpolation_aligned_spins_jax', 'inner_product', 'inner_product_jax', 'ann'
    psds : `dict`
        Dictionary of psds for different detectors. Default is None. If None, bilby's default psds will be used. Other options:\n
        Example 1: when values are psd name from pycbc analytical psds, psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044','V1':'AdvVirgo'}. To check available psd name run \n
        >>> import pycbc.psd
        >>> pycbc.psd.get_lalsim_psd_list()
        Example 2: when values are psd txt file available in bilby,
        psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt'}.
        For other psd files, check https://github.com/lscsoft/bilby/tree/master/bilby/gw/detector/noise_curves \n
        Example 3: when values are custom psd txt file. psds={'L1':'custom_psd.txt','H1':'custom_psd.txt'}. Custom created txt file has two columns. 1st column: frequency array, 2nd column: strain.
        Example 4: when you want psds to be created from a stretch of data for a given trigger time. psds={'L1':1246527224.169434} \n
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
        If True, it will set the maximum total mass of the binary according to the minimum frequency of the waveform. This is done searching for the maximum total mass corresponding to zero chirp time, i.e. the sytem merge below the minimum frequency. This is done to avoid unnecessary computation of SNR for systems that will not be detected. Default is False.
    pdet : `bool` or `str`
        If True or 'bool', calculate probability of detection using boolean method. If 'matched_filter', use matched filter probability. Default is False.
    snr_th : `float`
        SNR threshold for individual detector for pdet calculation. Default is 8.0.
    snr_th_net : `float`
        SNR threshold for network SNR for pdet calculation. Default is 8.0.
    ann_path_dict : `dict` or `str` or `None`
        Dictionary or path to json file containing ANN model and scaler paths for different detectors. Default is None (uses built-in models).
    snr_recalculation : `bool`
        If True, enables hybrid SNR recalculation for systems near detection threshold. Default is False.
    snr_recalculation_range : `list`
        SNR range [min, max] for triggering recalculation. Default is [6,8].
    snr_recalculation_waveform_approximant : `str`
        Waveform approximant to use for SNR recalculation. Default is 'IMRPhenomXPHM'.

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
    |:attr:`~spin_max`                    | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~mtot_resolution`             | `int`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~ratio_resolution`            | `int`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~spin_resolution`             | `int`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~ratio_arr`                   | `numpy.ndarray`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~mtot_arr`                    | `numpy.ndarray`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~a_1_arr`                     | `numpy.ndarray`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~a_2_arr`                     | `numpy.ndarray`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~sampling_frequency`          | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~waveform_approximant`        | `str`                            |
    +-------------------------------------+----------------------------------+
    |:attr:`~frequency_domain_source_model`| `str`                           |
    +-------------------------------------+----------------------------------+
    |:attr:`~f_min`                       | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~duration_max`                | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~duration_min`                | `float`                          |
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
    |:attr:`~snr_partialsacaled_list`     | `list` of `numpy.ndarray`        |
    +-------------------------------------+----------------------------------+
    |:attr:`~multiprocessing_verbose`     | `bool`                           |
    +-------------------------------------+----------------------------------+
    |:attr:`~param_dict_given`            | `dict`                           |
    +-------------------------------------+----------------------------------+
    |:attr:`~pdet`                        | `bool` or `str`                  |
    +-------------------------------------+----------------------------------+
    |:attr:`~snr_th`                      | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~snr_th_net`                  | `float`                          |
    +-------------------------------------+----------------------------------+
    |:attr:`~model_dict`                  | `dict` (ANN models)              |
    +-------------------------------------+----------------------------------+
    |:attr:`~scaler_dict`                 | `dict` (ANN scalers)             |
    +-------------------------------------+----------------------------------+
    |:attr:`~error_adjustment`            | `dict` (ANN error correction)    |
    +-------------------------------------+----------------------------------+
    |:attr:`~ann_catalogue`               | `dict` (ANN configuration)       |
    +-------------------------------------+----------------------------------+
    |:attr:`~snr_recalculation`           | `bool`                           |
    +-------------------------------------+----------------------------------+
    |:attr:`~snr_recalculation_range`     | `list`                           |
    +-------------------------------------+----------------------------------+
    |:attr:`~snr_recalculation_waveform_approximant`| `str`               |
    +-------------------------------------+----------------------------------+

    Instance Methods
    ----------
    GWSNR class has the following methods, \n
    +-------------------------------------+----------------------------------+
    | Methods                             | Description                      |
    +=====================================+==================================+
    |:meth:`~snr`                         | Main method that calls           |
    |                                     | appropriate SNR calculation      |
    |                                     | based on :attr:`~snr_type`.      |
    +-------------------------------------+----------------------------------+
    |:meth:`~snr_with_interpolation`      | Calculates SNR using             |
    |                                     | interpolation method.            |
    +-------------------------------------+----------------------------------+
    |:meth:`~snr_with_ann`                | Calculates SNR using             |
    |                                     | artificial neural network.       |
    +-------------------------------------+----------------------------------+
    |:meth:`~compute_bilby_snr`           | Calculates SNR using             |
    |                                     | inner product method             |
    |                                     | (python multiprocessing).        |
    +-------------------------------------+----------------------------------+
    |:meth:`~compute_ripple_snr`          | Calculates SNR using             |
    |                                     | inner product method             |
    |                                     | (jax.jit+jax.vmap).              |
    +-------------------------------------+----------------------------------+
    |:meth:`~detector_horizon`            | Calculates detector horizon      |
    |                                     | distance.                        |
    +-------------------------------------+----------------------------------+
    |:meth:`~probability_of_detection`    | Calculates probability of        |
    |                                     | detection.                       |
    +-------------------------------------+----------------------------------+
    |:meth:`~print_all_params`            | Prints all the parameters of     |
    |                                     | the class instance.              |
    +-------------------------------------+----------------------------------+
    |:meth:`~init_partialscaled`          | Generates partialscaled SNR      |
    |                                     | interpolation coefficients.      |
    +-------------------------------------+----------------------------------+
    |:meth:`~interpolator_setup`          | Sets up interpolator files       |
    |                                     | and handles caching.             |
    +-------------------------------------+----------------------------------+
    |:meth:`~ann_initilization`           | Initializes ANN models and       |
    |                                     | scalers for detection.           |
    +-------------------------------------+----------------------------------+
    |:meth:`~output_ann`                  | Prepares input features for      |
    |                                     | ANN prediction.                  |
    +-------------------------------------+----------------------------------+
    |:meth:`~calculate_mtot_max`          | Calculates maximum total mass    |
    |                                     | based on minimum frequency.      |
    +-------------------------------------+----------------------------------+
    """

    # Class attributes with documentation
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

    spin_max = None
    """``float`` \n
    Maximum spin magnitude for aligned-spin interpolation methods."""

    mtot_resolution = None
    """``int`` \n
    Number of points in the total mass array (use interpolation purpose)."""

    ratio_resolution = None
    """``int`` \n
    Number of points in the mass ratio array (use interpolation purpose)."""

    spin_resolution = None
    """``int`` \n
    Number of points in the spin arrays for aligned-spin interpolation methods."""

    ratio_arr = None
    """``numpy.ndarray`` \n
    Array of mass ratio."""

    mtot_arr = None
    """``numpy.ndarray`` \n
    Array of total mass."""

    a_1_arr = None
    """``numpy.ndarray`` \n
    Array of primary spin values for aligned-spin interpolation."""

    a_2_arr = None
    """``numpy.ndarray`` \n
    Array of secondary spin values for aligned-spin interpolation."""

    sampling_frequency = None
    """``float`` \n
    Sampling frequency of the detector."""

    waveform_approximant = None
    """``str`` \n
    Waveform approximant to use."""

    frequency_domain_source_model = None
    """``str`` \n
    Source model for frequency domain waveform generation."""

    f_min = None
    """``float`` \n
    Minimum frequency of the waveform."""

    duration_max = None
    """``float`` or ``None`` \n
    Maximum duration for waveform generation."""

    duration_min = None
    """``float`` or ``None`` \n
    Minimum duration for waveform generation."""

    snr_type = None
    """``str`` \n
    Type of SNR calculation. Options: 'interpolation', 'interpolation_no_spins', 'interpolation_no_spins_jax', 'interpolation_aligned_spins', 'interpolation_aligned_spins_jax', 'inner_product', 'inner_product_jax', 'ann'."""

    psds_list = None
    """``list`` of bilby's PowerSpectralDensity ``object`` \n
    List of power spectral density objects for different detectors."""

    detector_tensor_list = None
    """``list`` of detector tensor ``numpy.ndarray`` \n
    List of detector tensor arrays for antenna response calculations."""

    detector_list = None
    """``list`` of ``str`` \n
    List of detector names."""

    interpolator_dir = None
    """``str`` \n
    Path to store the interpolator pickle file."""

    path_interpolator = None
    """``list`` of ``str`` \n
    List of paths to interpolator pickle files for each detector."""

    snr_partialsacaled_list = None
    """``list`` of ``numpy.ndarray`` \n
    List of partial-scaled SNR interpolation coefficients for each detector."""

    multiprocessing_verbose = None
    """``bool`` \n
    If True, show progress bar during SNR computation with multiprocessing."""

    param_dict_given = None
    """``dict`` \n
    Dictionary containing interpolator parameters for identification and caching."""

    pdet = None
    """``bool`` or ``str`` \n
    If True or 'bool', calculate probability of detection using boolean method. If 'matched_filter', use matched filter probability. Default is False."""

    snr_th = None
    """``float`` \n
    SNR threshold for individual detector for pdet calculation. Default is 8.0."""

    snr_th_net = None
    """``float`` \n
    SNR threshold for network SNR for pdet calculation. Default is 8.0."""

    model_dict = None
    """``dict`` \n
    Dictionary of ANN models for different detectors (used when snr_type='ann')."""

    scaler_dict = None
    """``dict`` \n
    Dictionary of ANN feature scalers for different detectors (used when snr_type='ann')."""

    error_adjustment = None
    """``dict`` \n
    Dictionary of ANN error correction parameters for different detectors (used when snr_type='ann')."""

    ann_catalogue = None
    """``dict`` \n
    Dictionary containing ANN configuration and model paths (used when snr_type='ann')."""

    snr_recalculation = None
    """``bool`` \n
    If True, enables hybrid SNR recalculation for systems near detection threshold."""

    snr_recalculation_range = None
    """``list`` \n
    SNR range [min, max] for triggering recalculation."""

    snr_recalculation_waveform_approximant = None
    """``str`` \n
    Waveform approximant to use for SNR recalculation."""

    get_interpolated_snr = None
    """``function`` \n
    Function for interpolated SNR calculation (set based on snr_type)."""

    noise_weighted_inner_product_jax = None
    """``function`` \n
    JAX-accelerated noise-weighted inner product function (used when snr_type='inner_product_jax')."""

    def __init__(
        self,
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
        snr_recalculation_range=[4,12],
        snr_recalculation_waveform_approximant="IMRPhenomXPHM",
    ):
        """
        Initialize the GWSNR class for gravitational wave signal-to-noise ratio calculation.

        This method sets up the GWSNR instance with specified parameters for SNR computation,
        including detector configurations, waveform settings, and computational method selection.
        It automatically handles interpolator setup, detector PSD loading, and ANN model 
        initialization based on the chosen SNR calculation method.

        The initialization process includes:
        - Setting up detector configurations and power spectral densities
        - Configuring interpolation grids for fast SNR calculation (if applicable)
        - Loading or generating interpolation coefficients for partial-scaled SNR
        - Initializing ANN models and scalers for detection probability estimation
        - Validating parameter ranges and compatibility checks

        All parameters are documented in the class docstring above. This initialization
        method automatically calls appropriate setup routines based on the selected
        snr_type and prints configuration information if gwsnr_verbose is True.

        Raises
        ------
        ValueError
            If snr_type is not recognized or parameter combinations are invalid.
        FileNotFoundError
            If required interpolator files or ANN models cannot be found and cannot be generated.
        """

        print("\nInitializing GWSNR class...\n")
        # setting instance attributes
        self.npool = npool
        self.pdet = pdet
        self.snr_th = snr_th
        self.snr_th_net = snr_th_net
        self.duration_max = duration_max
        self.duration_min = duration_min
        self.snr_type = snr_type
        self.spin_max = spin_max

        # getting interpolator data from the package
        # first check if the interpolator directory './interpolator_pickle' exists
        if not pathlib.Path('./interpolator_pickle').exists():
            # Get the path to the resource
            with path('gwsnr.core', 'interpolator_pickle') as resource_path:
                print(f"Copying interpolator data from the library resource {resource_path} to the current working directory.")
                resource_path = pathlib.Path(resource_path)  # Ensure it's a Path object

                # Define destination path (same name in current working directory)
                dest_path = pathlib.Path.cwd() / interpolator_dir

                # Copy entire directory tree
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(resource_path, dest_path)

        # dealing with mtot_max
        # set max cut off according to minimum_frequency
        mtot_max = (
            mtot_max
            if not mtot_cut
            else self.calculate_mtot_max(mtot_max, minimum_frequency)
        )
        self.mtot_cut = mtot_cut
        self.mtot_max = mtot_max
        self.mtot_min = mtot_min
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.mtot_resolution = mtot_resolution
        self.ratio_resolution = ratio_resolution
        self.snr_recalculation = snr_recalculation
        if snr_recalculation:
            self.snr_recalculation = snr_recalculation
            self.snr_recalculation_range = snr_recalculation_range
            self.snr_recalculation_waveform_approximant = snr_recalculation_waveform_approximant

        self.ratio_arr = np.geomspace(ratio_min, ratio_max, ratio_resolution)
        self.mtot_arr = np.sort(mtot_min + mtot_max - np.geomspace(mtot_min, mtot_max, mtot_resolution))
        # buffer of 0.1 is added to the mtot
        # self.mtot_arr = np.concatenate((np.array([mtot_min-0.1]), np.sort(mtot_min + mtot_max - np.geomspace(mtot_min, mtot_max, mtot_resolution-2)), np.array([mtot_max+0.1])))

        self.sampling_frequency = sampling_frequency
        self.waveform_approximant = waveform_approximant
        self.frequency_domain_source_model = frequency_domain_source_model
        self.f_min = minimum_frequency
        self.interpolator_dir = interpolator_dir
        self.multiprocessing_verbose = multiprocessing_verbose

        self.spin_resolution = spin_resolution
        self.spin_max = spin_max
        self.a_1_arr = np.linspace(-self.spin_max, self.spin_max, self.spin_resolution)
        self.a_2_arr = np.linspace(-self.spin_max, self.spin_max, self.spin_resolution)

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
            self.duration_min = 4.0


        # now generate interpolator, if not exists
        if snr_type == "interpolation" or snr_type == "interpolation_no_spins" or snr_type == "interpolation_no_spins_jax":
            if snr_type == "interpolation_no_spins_jax":
                from ..jax import get_interpolated_snr_no_spins_jax
                self.get_interpolated_snr = get_interpolated_snr_no_spins_jax
            else:
                from ..numba import get_interpolated_snr_no_spins_numba
                self.get_interpolated_snr = get_interpolated_snr_no_spins_numba

            # dealing with interpolator
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

        elif snr_type == "interpolation_aligned_spins" or snr_type == "interpolation_aligned_spins_jax":

            if snr_type == "interpolation_aligned_spins_jax":
                from ..jax import get_interpolated_snr_aligned_spins_jax
                self.get_interpolated_snr = get_interpolated_snr_aligned_spins_jax
            else:
                from ..numba import get_interpolated_snr_aligned_spins_numba
                self.get_interpolated_snr = get_interpolated_snr_aligned_spins_numba

            self.param_dict_given['spin_max'] = self.spin_max
            self.param_dict_given['spin_resolution'] = self.spin_resolution
            # dealing with interpolator
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

        # inner product method doesn't need interpolator generation
        elif snr_type == "inner_product":
            pass
        
        # need to initialize RippleInnerProduct class
        elif snr_type == "inner_product_jax":
            from ..ripple import RippleInnerProduct

            ripple_class = RippleInnerProduct(
                waveform_name=waveform_approximant, 
                minimum_frequency=minimum_frequency, 
                sampling_frequency=sampling_frequency, 
                reference_frequency=minimum_frequency
                )

            self.noise_weighted_inner_product_jax = ripple_class.noise_weighted_inner_product_jax

        # ANN method still needs the partialscaledSNR interpolator.
        elif snr_type == "ann":

            from ..numba import get_interpolated_snr_aligned_spins_numba
            self.get_interpolated_snr = get_interpolated_snr_aligned_spins_numba
            # below is added to find the genereated interpolator path
            self.param_dict_given['spin_max'] = self.spin_max
            self.param_dict_given['spin_resolution'] = self.spin_resolution
            
            self.model_dict, self.scaler_dict, self.error_adjustment, self.ann_catalogue = self.ann_initilization(ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant, snr_th)
            # dealing with interpolator
            self.snr_type = "interpolation_aligned_spins"
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)
            self.snr_type = "ann"

        else:
            raise ValueError("SNR function type not recognised. Please choose from 'interpolation', 'interpolation_no_spins', 'interpolation_no_spins_jax', 'interpolation_aligned_spins', 'interpolation_aligned_spins_jax', 'inner_product', 'inner_product_jax', 'ann'.")

        # change back to original
        self.psds_list = psds_list
        self.detector_tensor_list = detector_tensor_list
        self.detector_list = detector_list

        if (snr_type == "inner_product") or (snr_type == "inner_product_jax"):
            self.snr_with_interpolation = self._print_no_interpolator

        # print some info
        self.print_all_params(gwsnr_verbose)
        print("\n")

    # dealing with interpolator
    def interpolator_setup(self, interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list):
        """
        Function to set up interpolator files and handle caching for partialscaled SNR interpolation coefficients.

        This method checks for existing interpolator files, determines which detectors need new interpolators,
        and manages the generation and loading of partialscaled SNR interpolation data. It handles both
        the creation of new interpolators and the loading of existing ones from cache.

        Parameters
        ----------
        interpolator_dir : `str`
            Path to directory for storing interpolator pickle files. Default is './interpolator_pickle'.
        create_new_interpolator : `bool`
            If True, forces generation of new interpolators or replaces existing ones. If False,
            uses existing interpolators when available. Default is False.
        psds_list : `list` of bilby's PowerSpectralDensity objects
            List of power spectral density objects for different detectors used for interpolator generation.
        detector_tensor_list : `list` of `numpy.ndarray`
            List of detector tensor arrays for antenna response calculations during interpolator generation.
        detector_list : `list` of `str`
            List of detector names (e.g., ['L1', 'H1', 'V1']) for which interpolators are needed.

        Returns
        -------
        path_interpolator_all : `list` of `str`
            List of file paths to partialscaled SNR interpolator pickle files for all detectors.
            These files contain the precomputed interpolation coefficients used for fast SNR calculation.

        Notes
        -----
        - The method uses :func:`~gwsnr.utils.interpolator_check` to determine which detectors need new interpolators
        - For missing interpolators, calls :meth:`~init_partialscaled` to generate them
        - Updates class attributes including :attr:`~psds_list`, :attr:`~detector_tensor_list`, :attr:`~detector_list`, and :attr:`~path_interpolator`
        - Loads all interpolator data into :attr:`~snr_partialsacaled_list` for runtime use
        - Supports both no-spin and aligned-spin interpolation methods based on :attr:`~snr_type`
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

        # len(detector_list) == 0, means all the detectors have interpolator stored
        if len(self.detector_list) > 0:
            print("Please be patient while the interpolator is generated")
            # if self.snr_type == 'interpolation_aligned_spins':
            #     self.init_partialscaled_aligned_spins()
            # else:
            self.init_partialscaled()
        elif create_new_interpolator:
            # change back to original
            self.psds_list = psds_list
            self.detector_tensor_list = detector_tensor_list
            self.detector_list = detector_list
            print("Please be patient while the interpolator is generated")
            # if self.snr_type == 'interpolation_aligned_spins':
            #     self.init_partialscaled_aligned_spins()
            # else:
            self.init_partialscaled()

        # get all partialscaledSNR from the stored interpolator
        self.snr_partialsacaled_list = np.array([
            load_pickle(path) for path in path_interpolator_all
        ], dtype=np.float64)

        return path_interpolator_all

    def ann_initilization(self, ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant, snr_th):
        """
        Function to initialize ANN models and scalers for detection probability estimation using artificial neural networks.

        This method loads and validates ANN models, feature scalers, and error correction parameters for each detector
        in the detector list. It handles both built-in models from the gwsnr package and user-provided models,
        ensuring compatibility with the current GWSNR configuration parameters.

        Parameters
        ----------
        ann_path_dict : `dict` or `str` or `None`
            Dictionary or path to JSON file containing ANN model and scaler paths for different detectors.
            If None, uses default models from gwsnr/ann/data/ann_path_dict.json.
            If dict, should have structure: {detector_name: {'model_path': str, 'scaler_path': str, 
            'error_adjustment_path': str, 'sampling_frequency': float, 'minimum_frequency': float, 
            'waveform_approximant': str, 'snr_th': float}}.
        detector_list : `list` of `str`
            List of detector names (e.g., ['L1', 'H1', 'V1']) for which ANN models are needed.
        sampling_frequency : `float`
            Sampling frequency of the detector data. Must match ANN training parameters.
        minimum_frequency : `float`
            Minimum frequency of the waveform. Must match ANN training parameters.
        waveform_approximant : `str`
            Waveform approximant to use. Must match ANN training parameters.
        snr_th : `float`
            SNR threshold for individual detector detection. Must match ANN training parameters.

        Returns
        -------
        model_dict : `dict`
            Dictionary of loaded ANN models for each detector {detector_name: tensorflow.keras.Model}.
        scaler_dict : `dict`
            Dictionary of loaded feature scalers for each detector {detector_name: sklearn.preprocessing.Scaler}.
        error_adjustment : `dict`
            Dictionary of error correction parameters for each detector {detector_name: {'slope': float, 'intercept': float}}.
        ann_catalogue : `dict`
            Dictionary containing complete ANN configuration and model paths for all detectors.

        Raises
        ------
        ValueError
            If ANN model or scaler is not available for a detector in detector_list.
            If model parameters don't match the current GWSNR configuration.
            If required keys ('model_path', 'scaler_path') are missing from ann_path_dict.

        Notes
        -----
        - Models are loaded from gwsnr/ann/data directory if paths don't exist as files
        - Parameter validation ensures ANN models are compatible with current settings
        - Error adjustment parameters provide post-prediction correction for improved accuracy
        - ANN models use partial-scaled SNR as input feature along with other parameters
        """

        # check the content ann_path_dict.json in gwsnr/ann module directory
        # e.g. ann_path_dict = dict(L1=dict(model_path='path_to_model', scaler_path='path_to_scaler', sampling_frequency=2048.0, minimum_frequency=20.0, waveform_approximant='IMRPhenomXPHM', snr_th=8.0))
        # there will be existing ANN model and scaler for default parameters

        # getting ann data from the package
        # first check if the ann_data directory './ann_data' exists
        if not pathlib.Path('./ann_data').exists():
            # Get the path to the resource
            with path('gwsnr.ann', 'ann_data') as resource_path:
                print(f"Copying ANN data from the library resource {resource_path} to the current working directory.")
                resource_path = pathlib.Path(resource_path)  # Ensure it's a Path object

                # Define destination path (same name in current working directory)
                dest_path = pathlib.Path.cwd() / resource_path.name

                # Copy entire directory tree
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(resource_path, dest_path)
 
        if ann_path_dict is None:
            print("ANN model and scaler path is not given. Using the default path.")
            ann_path_dict = './ann_data/ann_path_dict.json'
        else:
            print("ANN model and scaler path is given. Using the given path.")

        if isinstance(ann_path_dict, str):
            ann_path_dict = load_json(ann_path_dict)
        elif isinstance(ann_path_dict, dict):
            pass
        else:
            raise ValueError("ann_path_dict should be a dictionary or a path to the json file.")

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
                # load the model from gwsnr/ann/data directory
                model_dict[detector] = load_ann_h5_from_module('gwsnr', 'ann.data', ann_path_dict[detector]['model_path'])
                print(f"ANN model for {detector} is loaded from gwsnr/ann/data directory.")
            else:
                # load the model from the given path
                model_dict[detector] = load_ann_h5(ann_path_dict[detector]['model_path'])
                print(f"ANN model for {detector} is loaded from {ann_path_dict[detector]['model_path']}.")

            # get ann scaler
            if not os.path.exists(ann_path_dict[detector]['scaler_path']):
                # load the scaler from gwsnr/ann/data directory
                scaler_dict[detector] = load_pickle_from_module('gwsnr', 'ann.data', ann_path_dict[detector]['scaler_path'])
                print(f"ANN scaler for {detector} is loaded from gwsnr/ann/data directory.")
            else:
                # load the scaler from the given path
                scaler_dict[detector] = load_pickle(ann_path_dict[detector]['scaler_path'])
                print(f"ANN scaler for {detector} is loaded from {ann_path_dict[detector]['scaler_path']}.")

            # get error_adjustment
            if not os.path.exists(ann_path_dict[detector]['error_adjustment_path']):
                # load the error_adjustment from gwsnr/ann/data directory
                error_adjustment[detector] = load_json_from_module('gwsnr', 'ann.data', ann_path_dict[detector]['error_adjustment_path'])
                print(f"ANN error_adjustment for {detector} is loaded from gwsnr/ann/data directory.")
            else:
                # load the error_adjustment from the given path
                error_adjustment[detector] = load_json(ann_path_dict[detector]['error_adjustment_path'])
                print(f"ANN error_adjustment for {detector} is loaded from {ann_path_dict[detector]['error_adjustment_path']}.")

        return model_dict, scaler_dict, error_adjustment, ann_path_dict

    def _print_no_interpolator(self, **kwargs):
        """
        Function to print error message when no interpolator is found for interpolation-based SNR calculation.

        This is a placeholder method that is assigned to :meth:`~snr_with_interpolation` when :attr:`~snr_type` 
        is set to 'inner_product' or 'inner_product_jax', indicating that interpolation functionality 
        is not available for these SNR calculation methods.

        Parameters
        ----------
        **kwargs : `dict`
            Arbitrary keyword arguments passed to the method (ignored).

        Raises
        ------
        ValueError
            Always raises an error indicating that no interpolator is available and suggesting 
            to use snr_type="interpolation" to generate interpolators.

        Notes
        -----
        - This method is automatically assigned during class initialization when using inner product methods
        - Serves as a safety mechanism to prevent calls to interpolation methods when unavailable
        - Users should initialize GWSNR with appropriate interpolation snr_type to access interpolation functionality
        """

        raise ValueError(
            'No interpolator found. Please set snr_type="interpolation" to generate new interpolator.'
        )

    def calculate_mtot_max(self, mtot_max, minimum_frequency):
        """
        Function to calculate the maximum total mass cutoff based on minimum frequency to ensure positive chirp time.

        This method determines the maximum allowable total mass for binary systems by finding where 
        the chirp time becomes zero at the given minimum frequency. The chirp time represents the 
        duration a gravitational wave signal spends in the detector's frequency band. A safety factor 
        of 1.1 is applied to ensure the chirp time remains positive for waveform generation.

        The calculation uses the :func:`~gwsnr.numba.findchirp_chirptime` function to compute chirp 
        times and employs numerical root finding to determine where the chirp time approaches zero.

        Parameters
        ----------
        mtot_max : `float`
            User-specified maximum total mass of the binary in solar masses. If this exceeds 
            the frequency-based limit, it will be reduced to the calculated maximum.
        minimum_frequency : `float`
            Minimum frequency of the waveform in Hz. Lower frequencies allow higher total masses 
            before the chirp time becomes negative.

        Returns
        -------
        mtot_max : `float`
            Adjusted maximum total mass of the binary in solar masses, ensuring positive chirp 
            time at the given minimum frequency. Will be the smaller of the input mtot_max and 
            the frequency-based limit.

        Notes
        -----
        - Uses equal mass ratio (q=1.0) for the chirp time calculation as a conservative estimate
        - The safety factor of 1.1 provides a buffer to prevent numerical issues during waveform generation
        - This limit is particularly important for low-frequency detectors and TaylorF2 approximants
        - The method uses :func:`scipy.optimize.fsolve` to find the root of the chirp time function
        """

        def func(x, mass_ratio=1.0):
            mass_1 = x / (1 + mass_ratio)
            mass_2 = x / (1 + mass_ratio) * mass_ratio

            return findchirp_chirptime(mass_1, mass_2, minimum_frequency)

        # find where func is zero
        from scipy.optimize import fsolve
        
        mtot_max_generated = fsolve(func, 150)[
            0
        ]  # to make sure that chirptime is not negative, TaylorF2 might need this
        if mtot_max > mtot_max_generated:
            mtot_max = mtot_max_generated

        return mtot_max

    def print_all_params(self, verbose=True):
        """
        Function to print all the parameters and configuration of the GWSNR class instance.

        This method displays comprehensive information about the current GWSNR configuration including
        computational parameters, waveform settings, detector configuration, interpolation grid parameters,
        and file paths. It provides a complete overview of the initialized GWSNR instance for verification
        and debugging purposes.

        Parameters
        ----------
        verbose : `bool`
            If True, print all the parameters of the class instance to stdout. If False,
            suppress output. Default is True.

        Notes
        -----
        The printed information includes:
        
        - **Computational settings**: Number of processors (:attr:`~npool`), SNR calculation type (:attr:`~snr_type`)
        - **Waveform configuration**: Approximant (:attr:`~waveform_approximant`), sampling frequency (:attr:`~sampling_frequency`), minimum frequency (:attr:`~f_min`)
        - **Mass parameter ranges**: Total mass bounds (:attr:`~mtot_min`, :attr:`~mtot_max`) with frequency-based cutoff information
        - **Detector setup**: List of detectors (:attr:`~detector_list`) and their PSDs (:attr:`~psds_list`)
        - **Interpolation parameters**: Mass ratio bounds (:attr:`~ratio_min`, :attr:`~ratio_max`), grid resolutions (:attr:`~mtot_resolution`, :attr:`~ratio_resolution`)
        - **File paths**: Interpolator directory (:attr:`~interpolator_dir`) when using interpolation methods

        This method is automatically called during class initialization when :attr:`~gwsnr_verbose` is True.

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> snr = GWSNR(gwsnr_verbose=False)  # Initialize without printing
        >>> snr.print_all_params()  # Manually print parameters
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
        Main function to calculate SNR of gravitational-wave signals from compact binary coalescences.

        This method serves as the primary interface for SNR calculation, automatically routing to the 
        appropriate computation method based on the :attr:`~snr_type` setting. It supports multiple 
        backend methods including interpolation-based fast calculation, inner product methods, JAX-accelerated 
        computation, and artificial neural network estimation.

        The method handles parameter validation, coordinate transformations (e.g., tilt angles to aligned spins), 
        and optionally computes probability of detection. For systems near detection threshold, it can perform 
        hybrid SNR recalculation using more accurate waveform models.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar masses. Default is np.array([10.0]).
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar masses. Default is np.array([10.0]).
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Gravitational wave polarization angle in radians. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Gravitational wave phase at coalescence in radians. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the source in radians. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the source in radians. Default is 0.0.
        a_1 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the primary object. Default is 0.0.
        a_2 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the secondary object. Default is 0.0.
        tilt_1 : `numpy.ndarray` or `float`
            Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.
        tilt_2 : `numpy.ndarray` or `float`
            Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.
        phi_12 : `numpy.ndarray` or `float`
            Azimuthal angle between the two spins in radians. Default is 0.0.
        phi_jl : `numpy.ndarray` or `float`
            Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.
        lambda_1 : `numpy.ndarray` or `float`
            Dimensionless tidal deformability of primary object. Default is 0.0.
        lambda_2 : `numpy.ndarray` or `float`
            Dimensionless tidal deformability of secondary object. Default is 0.0.
        eccentricity : `numpy.ndarray` or `float`
            Orbital eccentricity at reference frequency. Default is 0.0.
        gw_param_dict : `dict` or `bool`
            Dictionary containing all gravitational wave parameters as key-value pairs. 
            If provided, takes precedence over individual parameter arguments. Default is False.
        output_jsonfile : `str` or `bool`
            If string, saves the SNR results to a JSON file with the given filename. 
            If True, saves to 'snr.json'. If False, no file output. Default is False.

        Returns
        -------
        snr_dict : `dict`
            Dictionary containing SNR values for each detector and network SNR.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'optimal_snr_net'.
            Values are numpy arrays of SNR values corresponding to input parameters.
        pdet_dict : `dict`
            Dictionary containing probability of detection values (only if :attr:`~pdet` is True).
            Keys include detector names and 'pdet_net'. Values are numpy arrays of probabilities.

        Raises
        ------
        ValueError
            If :attr:`~snr_type` is not recognized or if parameters are outside valid ranges.

        Notes
        -----
        - For interpolation methods, aligned spin components are computed as a_i * cos(tilt_i)
        - Total mass must be within [mtot_min, mtot_max] range for interpolation methods
        - Hybrid SNR recalculation is triggered when :attr:`~snr_recalculation` is True and 
          network SNR falls within :attr:`~snr_recalculation_range`
        - When :attr:`~pdet` is True, returns detection probabilities instead of SNR values

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> # Basic interpolation-based SNR calculation
        >>> snr = GWSNR(snr_type='interpolation')
        >>> result = snr.snr(mass_1=30.0, mass_2=30.0, luminosity_distance=100.0)
        
        >>> # Using parameter dictionary
        >>> params = {'mass_1': [20, 30], 'mass_2': [20, 30], 'luminosity_distance': [100, 200]}
        >>> result = snr.snr(gw_param_dict=params)
        
        >>> # With probability of detection
        >>> snr_pdet = GWSNR(snr_type='interpolation', pdet=True)
        >>> pdet_result = snr_pdet.snr(mass_1=30.0, mass_2=30.0, luminosity_distance=100.0)
        """

        if not gw_param_dict:
            gw_param_dict = {
                "mass_1": mass_1,
                "mass_2": mass_2,
                "luminosity_distance": luminosity_distance,
                "theta_jn": theta_jn,
                "psi": psi,
                "phase": phase,
                "geocent_time": geocent_time,
                "ra": ra,
                "dec": dec,
                "a_1": a_1,
                "a_2": a_2,
                "tilt_1": tilt_1,
                "tilt_2": tilt_2,
                "phi_12": phi_12,
                "phi_jl": phi_jl,
                "lambda_1": lambda_1,
                "lambda_2": lambda_2,
                "eccentricity": eccentricity
            }

        interpolation_list = [
            "interpolation",
            "interpolation_no_spins",
            "interpolation_aligned_spins",
            "interpolation_no_spins_jax",
            "interpolation_aligned_spins_jax",
        ]

        if self.snr_type in interpolation_list:

            # if tilt_1, tilt_2 are given, 
            # then a_1 = a_1 * np.cos(tilt_1)
            # a_2 = a_2 * np.cos(tilt_2)
            # first check if a_1 and a_2 are not less than 0.0
            # if tilt_1 and tilt_2 is in gw_param_dict
            tilt_1 = gw_param_dict.get("tilt_1", tilt_1)
            tilt_2 = gw_param_dict.get("tilt_2", tilt_2)
            a_1 = gw_param_dict.get("a_1", a_1)
            a_2 = gw_param_dict.get("a_2", a_2)
            a_1_old = a_1
            a_2_old = a_2
            # if tilt_1 and tilt_2 numpy.ndarray or list, convert them to numpy array
            if isinstance(tilt_1, (list, np.ndarray)):
                a_1 = np.array(a_1, ndmin=1)
                a_2 = np.array(a_2, ndmin=1)
                tilt_1 = np.array(tilt_1, ndmin=1)
                tilt_2 = np.array(tilt_2, ndmin=1)
                a_1 = a_1 * np.cos(tilt_1)
                a_2 = a_2 * np.cos(tilt_2)
                if gw_param_dict:
                    gw_param_dict["a_1"] = a_1
                    gw_param_dict["a_2"] = a_2

            print("solving SNR with interpolation")
            snr_dict = self.snr_with_interpolation(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )
            gw_param_dict['a_1'] = a_1_old
            gw_param_dict['a_2'] = a_2_old

        elif self.snr_type == "inner_product":
            print("solving SNR with inner product")

            snr_dict = self.compute_bilby_snr(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )

        elif self.snr_type == "inner_product_jax":
            print("solving SNR with inner product JAX")

            snr_dict = self.compute_ripple_snr(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )

        elif self.snr_type == "ann":
            snr_dict = self.snr_with_ann(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )

        else:
            raise ValueError("SNR function type not recognised")
        

        if self.snr_recalculation:
            waveform_approximant_old = self.waveform_approximant
            self.waveform_approximant = self.snr_recalculation_waveform_approximant

            optimal_snr_net = snr_dict["optimal_snr_net"]
            min_, max_ = self.snr_recalculation_range
            idx = np.logical_and(
                optimal_snr_net >= min_,
                optimal_snr_net <= max_,
            )
            if np.sum(idx) != 0:
                print(
                    f"Recalculating SNR for {np.sum(idx)} out of {len(optimal_snr_net)} samples in the SNR range of {min_} to {max_}"
                )
                # print(f'\n length of idx: {len(idx)}')
                # print(f'\n length of tilt_2: {len(idx)}')
                input_dict = {}
                for key in gw_param_dict.keys():
                    input_dict[key] = gw_param_dict[key][idx]

                snr_dict_ = self.compute_bilby_snr(
                    gw_param_dict=input_dict,
                )

                # iterate over detectors and update the snr_dict
                for key in snr_dict.keys():
                    if key in snr_dict_.keys():
                        snr_dict[key][idx] = snr_dict_[key]

            self.waveform_approximant = waveform_approximant_old
        

        if self.pdet:
            pdet_dict = self.probability_of_detection(snr_dict=snr_dict, snr_th=8.0, snr_th_net=8.0, type=self.pdet)

            return pdet_dict
        else:
            return snr_dict

    def snr_with_ann(
        self,
        mass_1=30.,
        mass_2=29.,
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
        Function to calculate SNR using artificial neural network (ANN) estimation method.

        This method uses trained neural networks to rapidly estimate the probability of detection (Pdet) 
        for spin-precessing gravitational wave signals. The ANN models leverage partial-scaled SNR as a 
        summary statistic along with other intrinsic parameters to provide fast detection probability 
        estimates, particularly useful for population synthesis studies.

        The method first calculates partial-scaled SNR using interpolation, then uses this as input 
        to pre-trained ANN models for each detector. Error correction is applied to improve accuracy
        of the ANN predictions.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar masses. Default is 30.0.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar masses. Default is 29.0.
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Gravitational wave polarization angle in radians. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Gravitational wave phase at coalescence in radians. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the source in radians. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the source in radians. Default is 0.0.
        a_1 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the primary object. Default is 0.0.
        a_2 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the secondary object. Default is 0.0.
        tilt_1 : `numpy.ndarray` or `float`
            Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.
        tilt_2 : `numpy.ndarray` or `float`
            Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.
        phi_12 : `numpy.ndarray` or `float`
            Azimuthal angle between the two spins in radians. Default is 0.0.
        phi_jl : `numpy.ndarray` or `float`
            Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.
        gw_param_dict : `dict` or `bool`
            Dictionary containing all gravitational wave parameters as key-value pairs. 
            If provided, takes precedence over individual parameter arguments. Default is False.
        output_jsonfile : `str` or `bool`
            If string, saves the SNR results to a JSON file with the given filename. 
            If True, saves to 'snr.json'. If False, no file output. Default is False.

        Returns
        -------
        optimal_snr : `dict`
            Dictionary containing ANN-estimated SNR values for each detector and network SNR.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'optimal_snr_net'.
            Values are numpy arrays of SNR estimates corresponding to input parameters.

        Raises
        ------
        ValueError
            If total mass (mass_1 + mass_2) is outside the range [mtot_min, mtot_max].

        Notes
        -----
        - ANN models must be pre-trained and loaded during class initialization
        - Uses aligned spin components calculated as a_i * cos(tilt_i) for feature input
        - Feature inputs include: partial-scaled SNR, amplitude factor, symmetric mass ratio, 
          effective spin, and inclination angle
        - Error adjustment parameters provide post-prediction correction for improved accuracy
        - Compatible with waveform approximants that have corresponding trained ANN models
        - Requires :attr:`~snr_type` to be set to 'ann' during GWSNR initialization

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> # Initialize with ANN method
        >>> snr = GWSNR(snr_type='ann', waveform_approximant='IMRPhenomXPHM')
        >>> # Calculate SNR using ANN
        >>> result = snr.snr_with_ann(mass_1=30.0, mass_2=25.0, luminosity_distance=200.0, 
        ...                          a_1=0.5, a_2=0.3, tilt_1=0.2, tilt_2=0.1)
        
        >>> # Using parameter dictionary
        >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200],
        ...           'a_1': [0.2, 0.5], 'tilt_1': [0.1, 0.3]}
        >>> result = snr.snr_with_ann(gw_param_dict=params)
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
            # adjusting the optimal SNR with error adjustment
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
        Function to prepare input features for ANN prediction from gravitational wave parameters.

        This method transforms gravitational wave parameters into feature vectors suitable for 
        artificial neural network prediction of detection probabilities. It calculates partial-scaled 
        SNR using interpolation and combines it with other intrinsic parameters to create the input 
        features expected by the pre-trained ANN models.

        The feature vector for each detector includes:
        - Partial-scaled SNR (dimensionless, distance-independent)
        - Amplitude factor (A1 = Mc^(5/6) / d_eff)
        - Symmetric mass ratio (eta)
        - Effective spin (chi_eff)
        - Inclination angle (theta_jn)

        Parameters
        ----------
        idx : `numpy.ndarray` of `bool`
            Boolean index array indicating which parameter entries are within valid mass ranges 
            for interpolation (mtot_min <= mtot <= mtot_max).
        params : `dict`
            Dictionary containing gravitational wave parameters with keys:
            - 'mass_1', 'mass_2': Primary and secondary masses in solar masses
            - 'luminosity_distance': Distance in Mpc
            - 'theta_jn': Inclination angle in radians
            - 'a_1', 'a_2': Spin magnitudes (dimensionless)
            - 'tilt_1', 'tilt_2': Spin tilt angles in radians
            - 'psi', 'geocent_time', 'ra', 'dec': Extrinsic parameters

        Returns
        -------
        ann_input : `list` of `numpy.ndarray`
            List of feature arrays for each detector in :attr:`~detector_list`.
            Each array has shape (N, 5) where N is the number of valid parameter sets,
            and columns correspond to [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn].

        Notes
        -----
        - Uses :meth:`~get_interpolated_snr` to calculate partial-scaled SNR via interpolation
        - Aligned spin components are computed as a_i * cos(tilt_i) for chi_eff calculation
        - Chirp mass Mc = (m1*m2)^(3/5) / (m1+m2)^(1/5) is used for amplitude scaling
        - Effective spin chi_eff = (m1*a1z + m2*a2z) / (m1+m2) where aiz are aligned components
        - Feature scaling is applied later using pre-loaded scalers in :meth:`~snr_with_ann`
        """

        mass_1 = np.array(params['mass_1'][idx])
        mass_2 = np.array(params['mass_2'][idx])
        luminosity_distance = np.array(params['luminosity_distance'][idx])
        theta_jn = np.array(params['theta_jn'][idx])
        psi = np.array(params['psi'][idx])
        geocent_time = np.array(params['geocent_time'][idx])
        ra = np.array(params['ra'][idx])
        dec = np.array(params['dec'][idx])
        a_1 = np.array(params['a_1'][idx])
        a_2 = np.array(params['a_2'][idx])
        tilt_1 = np.array(params['tilt_1'][idx])
        tilt_2 = np.array(params['tilt_2'][idx])
        # effective spin
        chi_eff = (mass_1 * a_1 * np.cos(tilt_1) + mass_2 * a_2 * np.cos(tilt_2)) / (mass_1 + mass_2)

        # to get the components of the spin aligned with angular momentum
        a_1 = a_1 * np.cos(tilt_1)
        a_2 = a_2 * np.cos(tilt_2)

        _, _, snr_partial, d_eff = self.get_interpolated_snr(
            np.array(mass_1),
            np.array(mass_2),
            np.array(luminosity_distance),
            np.array(theta_jn),
            np.array(psi),
            np.array(geocent_time),
            np.array(ra),
            np.array(dec),
            np.array(a_1),
            np.array(a_2),
            np.array(self.detector_tensor_list),
            np.array(self.snr_partialsacaled_list),
            np.array(self.ratio_arr),
            np.array(self.mtot_arr),
            np.array(self.a_1_arr),
            np.array(self.a_2_arr),
        )

        # calculate the effective amplitude
        Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
        eta = mass_1 * mass_2/(mass_1 + mass_2)**2.
        A1 = Mc ** (5.0 / 6.0)
        amp0 = A1 / np.array(d_eff)
        # inclination angle
        theta_jn = np.array(params["theta_jn"][idx])

        snr_partial = np.array(snr_partial)
        # for the detectors
        ann_input = []
        for i in range(len(self.detector_list)):
            ann_input.append(
                np.vstack([snr_partial[i], amp0[i], eta, chi_eff, theta_jn]).T
            )

        return (ann_input)

    def snr_with_interpolation(
        self,
        mass_1=30.,
        mass_2=29.,
        luminosity_distance=100.0,
        theta_jn=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=1246527224.169434,
        ra=0.0,
        dec=0.0,
        a_1=0.0,
        a_2=0.0,
        output_jsonfile=False,
        gw_param_dict=False,
    ):
        """
        Function to calculate SNR using bicubic interpolation of precomputed partial-scaled SNR coefficients.

        This method provides fast SNR calculation by interpolating precomputed partial-scaled SNR values
        across a grid of intrinsic parameters (total mass, mass ratio, and optionally aligned spins).
        The interpolation is performed using either Numba-accelerated or JAX-accelerated functions
        depending on the :attr:`~snr_type` setting. This approach is particularly efficient for
        large-scale population studies and parameter estimation.

        The method handles parameter validation, ensures masses are within interpolation bounds,
        and computes antenna response patterns for each detector. For systems outside the mass
        range, SNR values are set to zero.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar masses. Default is 30.0.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar masses. Default is 29.0.
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Gravitational wave polarization angle in radians. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Gravitational wave phase at coalescence in radians. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the source in radians. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the source in radians. Default is 0.0.
        a_1 : `numpy.ndarray` or `float`
            Dimensionless aligned spin component of the primary object (only used for aligned-spin interpolation types). Default is 0.0.
        a_2 : `numpy.ndarray` or `float`
            Dimensionless aligned spin component of the secondary object (only used for aligned-spin interpolation types). Default is 0.0.
        gw_param_dict : `dict` or `bool`
            Dictionary containing all gravitational wave parameters as key-value pairs. 
            If provided, takes precedence over individual parameter arguments. Default is False.
        output_jsonfile : `str` or `bool`
            If string, saves the SNR results to a JSON file with the given filename. 
            If True, saves to 'snr.json'. If False, no file output. Default is False.

        Returns
        -------
        optimal_snr : `dict`
            Dictionary containing SNR values for each detector and network SNR.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'optimal_snr_net'.
            Values are numpy arrays of SNR values corresponding to input parameters.
            Systems with total mass outside [mtot_min, mtot_max] have SNR set to zero.

        Notes
        -----
        - Requires precomputed interpolation coefficients stored in :attr:`~snr_partialsacaled_list`
        - Total mass (mass_1 + mass_2) must be within [mtot_min, mtot_max] for non-zero SNR
        - For aligned-spin methods, uses aligned spin components computed as a_i * cos(tilt_i)
        - Interpolation grid parameters are set during class initialization
        - Supports both Numba and JAX backends for accelerated computation
        - Compatible with waveform approximants: IMRPhenomD, TaylorF2, IMRPhenomXPHM

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> # No-spin interpolation
        >>> snr = GWSNR(snr_type='interpolation_no_spins')
        >>> result = snr.snr_with_interpolation(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
        
        >>> # Aligned-spin interpolation
        >>> snr_spin = GWSNR(snr_type='interpolation_aligned_spins')
        >>> result = snr_spin.snr_with_interpolation(mass_1=30.0, mass_2=25.0, 
        ...                                         luminosity_distance=100.0, a_1=0.5, a_2=0.3)
        
        >>> # Using parameter dictionary
        >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
        >>> result = snr.snr_with_interpolation(gw_param_dict=params)
        """

        # getting the parameters from the dictionary
        if gw_param_dict is not False:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, _, _, _, _, _, _, _ = get_gw_parameters(gw_param_dict)
        else:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, _, _, _, _, _, _, _ = get_gw_parameters(dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, ra=ra, dec=dec, a_1=a_1, a_2=a_2))

        # setting up the parameters
        detector_tensor = np.array(self.detector_tensor_list)
        detectors = np.array(self.detector_list)
        snr_partialscaled = np.array(self.snr_partialsacaled_list)

        size = len(mass_1)
        mtot = mass_1 + mass_2
        # Check if mtot is within the range of mtot_min and mtot_max
        idx2 = np.logical_and(mtot >= self.mtot_min, mtot <= self.mtot_max)
        idx_tracker = np.nonzero(idx2)[0]

        # Set multiprocessing start method to 'spawn' for multri-threading compatibility
        # mp.set_start_method('spawn', force=True)

        # Get interpolated SNR
        if self.snr_type == "interpolation" or self.snr_type == "interpolation_no_spins" or self.snr_type == "interpolation_aligned_spins":
            
            snr, snr_effective, _, _ = self.get_interpolated_snr(
                np.array(mass_1[idx2], dtype=np.float64),
                np.array(mass_2[idx2], dtype=np.float64),
                np.array(luminosity_distance[idx2], dtype=np.float64),
                np.array(theta_jn[idx2], dtype=np.float64),
                np.array(psi[idx2], dtype=np.float64),
                np.array(geocent_time[idx2], dtype=np.float64),
                np.array(ra[idx2], dtype=np.float64),
                np.array(dec[idx2], dtype=np.float64),
                np.array(a_1[idx2], dtype=np.float64),
                np.array(a_2[idx2], dtype=np.float64),
                np.array(detector_tensor, dtype=np.float64),
                np.array(snr_partialscaled, dtype=np.float64),
                np.array(self.ratio_arr, dtype=np.float64),
                np.array(self.mtot_arr, dtype=np.float64),
                np.array(self.a_1_arr, dtype=np.float64),
                np.array(self.a_2_arr, dtype=np.float64),
            )
        elif self.snr_type == "interpolation_no_spins_jax" or self.snr_type == "interpolation_aligned_spins_jax":
            import jax
            import jax.numpy as jnp
            jax.config.update("jax_enable_x64", True)

            snr, snr_effective, _, _ = self.get_interpolated_snr(
                jnp.array(mass_1[idx2]),
                jnp.array(mass_2[idx2]),
                jnp.array(luminosity_distance[idx2]),
                jnp.array(theta_jn[idx2]),
                jnp.array(psi[idx2]),
                jnp.array(geocent_time[idx2]),
                jnp.array(ra[idx2]),
                jnp.array(dec[idx2]),
                jnp.array(a_1[idx2]),
                jnp.array(a_2[idx2]),
                jnp.array(detector_tensor),
                jnp.array(snr_partialscaled),
                jnp.array(self.ratio_arr),
                jnp.array(self.mtot_arr),
                jnp.array(self.a_1_arr),
                jnp.array(self.a_2_arr),
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
        Function to generate partialscaled SNR interpolation coefficients for fast bicubic interpolation.

        This method computes and saves precomputed partial-scaled SNR values across a grid of intrinsic 
        parameters (total mass, mass ratio, and optionally aligned spins) for each detector in the network. 
        The partial-scaled SNR is distance-independent and decoupled from extrinsic parameters, enabling 
        fast interpolation during runtime SNR calculations.

        The method creates a parameter grid based on the interpolation type:
        - For no-spin methods: 2D grid over (mass_ratio, total_mass)
        - For aligned-spin methods: 4D grid over (mass_ratio, total_mass, a_1, a_2)

        For each grid point, it computes the optimal SNR using :meth:`~compute_bilby_snr` with fixed 
        extrinsic parameters, then scales by effective luminosity distance and chirp mass to create 
        the partial-scaled SNR coefficients. These coefficients are saved as pickle files for later 
        use during interpolation-based SNR calculations.

        Parameters
        ----------
        None
            Uses class attributes for grid parameters and detector configuration.

        Returns
        -------
        None
            Saves interpolation coefficients to pickle files specified in :attr:`~path_interpolator`.

        Raises
        ------
        ValueError
            If :attr:`~mtot_min` is less than 1.0 solar mass.
            If :attr:`~snr_type` is not supported for interpolation.

        Notes
        -----
        - Uses fixed extrinsic parameters: luminosity_distance=100 Mpc, theta_jn=0, ra=0, dec=0, psi=0, phase=0
        - Calls :meth:`~compute_bilby_snr` to generate unscaled SNR values across the parameter grid
        - Partial-scaled SNR = (optimal_SNR * d_eff) / Mc^(5/6) where Mc is chirp mass
        - Grid dimensions depend on resolution parameters: :attr:`~ratio_resolution`, :attr:`~mtot_resolution`, :attr:`~spin_resolution`
        - For aligned-spin methods, grid covers spin range [-spin_max, +spin_max] for both objects
        - Interpolation coefficients enable fast runtime SNR calculation via bicubic interpolation
        - Compatible with snr_types: 'interpolation', 'interpolation_no_spins', 'interpolation_aligned_spins', and their JAX variants

        Examples
        --------
        This method is called automatically during GWSNR initialization when interpolation 
        coefficients don't exist or when :attr:`~create_new_interpolator` is True.

        >>> from gwsnr import GWSNR
        >>> # Will automatically call init_partialscaled() if needed
        >>> snr = GWSNR(snr_type='interpolation_no_spins', create_new_interpolator=True)
        """

        if self.mtot_min < 1.0:
            raise ValueError("Error: mass too low")
        
        detectors = self.detector_list.copy()
        detector_tensor = self.detector_tensor_list.copy()
        num_det = np.arange(len(detectors), dtype=int)
        mtot_table = self.mtot_arr.copy()
        ratio_table = self.ratio_arr.copy()
        size1 = self.ratio_resolution
        size2 = self.mtot_resolution  

        # Assume these are 1D arrays with correct lengths
        ratio_table = np.asarray(ratio_table)
        mtot_table = np.asarray(mtot_table)

        list_1 = ["interpolation_aligned_spins", "interpolation_aligned_spins_numba", "interpolation_aligned_spins_jax"]
        list_2 = ["interpolation", "interpolation_no_spins", "interpolation_no_spins_numba", "interpolation_no_spins_jax"]
        
        # Create broadcastable 4D grids
        if self.snr_type in list_1:
            a_1_table = self.a_1_arr.copy()
            a_2_table = self.a_2_arr.copy()
            size3 = self.spin_resolution
            size4 = self.spin_resolution
            a_1_table = np.asarray(a_1_table)
            a_2_table = np.asarray(a_2_table)

            q, mtot, a_1, a_2 = np.meshgrid(
                ratio_table, mtot_table, a_1_table, a_2_table, indexing='ij'
            )
        elif self.snr_type  in list_2:
            q, mtot = np.meshgrid(ratio_table, mtot_table, indexing='ij')
            a_1 = np.zeros_like(mtot)
            a_2 = a_1

        mass_1 = mtot / (1 + q)
        mass_2 = mass_1 * q

        # geocent_time cannot be array here
        # this geocent_time is only to get partialScaledSNR
        geocent_time_ = 1246527224.169434  # random time from O3
        theta_jn_, ra_, dec_, psi_, phase_ = np.zeros(5)
        luminosity_distance_ = 100.0

        # Vectorized computation for effective luminosity distance
        Fp = np.array(
            [
                antenna_response_plus(ra_, dec_, geocent_time_, psi_, tensor)
                for tensor in detector_tensor
            ]
        )
        Fc = np.array(
            [
                antenna_response_cross(ra_, dec_, geocent_time_, psi_, tensor,)
                for tensor in detector_tensor
            ]
        )
        dl_eff = luminosity_distance_ / np.sqrt(
            Fp**2 * ((1 + np.cos(theta_jn_) ** 2) / 2) ** 2
            + Fc**2 * np.cos(theta_jn_) ** 2
        )

        print(f"Generating interpolator for {detectors} detectors")
        # calling bilby_snr
        optimal_snr_unscaled = self.compute_bilby_snr(
            mass_1=mass_1.flatten(),
            mass_2=mass_2.flatten(),
            luminosity_distance=luminosity_distance_,
            theta_jn=theta_jn_,
            psi=psi_,
            phase=phase_,
            geocent_time=geocent_time_,
            ra=ra_,
            dec=dec_,
            a_1=a_1.flatten(),
            a_2=a_2.flatten(),
        )

        # for partialscaledSNR
        Mchirp = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5)) # shape (size1, size2, size3, size4)
        Mchirp_scaled = Mchirp ** (5.0 / 6.0)
        # filling in interpolation table for different detectors
        list_1 = ["interpolation_aligned_spins", "interpolation_aligned_spins_numba", "interpolation_aligned_spins_jax"]
        list_2 = ["interpolation", "interpolation_no_spins", "interpolation_no_spins_numba", "interpolation_no_spins_jax"]

        for j in num_det:
            if self.snr_type in list_1:
                snr_partial_ = np.array(np.reshape(optimal_snr_unscaled[detectors[j]],(size1, size2, size3, size4)) * dl_eff[j] / Mchirp_scaled, dtype=np.float32), # shape (size1, size2, size3, size4)
            elif self.snr_type in list_2:
                snr_partial_ = np.array(np.reshape(optimal_snr_unscaled[detectors[j]],(size1, size2)) * dl_eff[j] / Mchirp_scaled, dtype=np.float32), # shape (size1, size2, size3, size4)
            else:
                raise ValueError(f"snr_type {self.snr_type} is not supported for interpolation.")
            # print('dl_eff=',dl_eff[j])
            # print('Mchirp_scaled=',Mchirp_scaled.shape)
            # print('optimal_snr_unscaled=',np.reshape(optimal_snr_unscaled[detectors[j]],(size1, size2, size3, size4)).shape)
            print(f"\nSaving Partial-SNR for {detectors[j]} detector with shape {snr_partial_[0].shape}")
            save_pickle(self.path_interpolator[j], snr_partial_[0])

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
        Function to calculate SNR using noise-weighted inner product method with LAL waveform generation.

        This method computes the optimal signal-to-noise ratio using the standard matched filtering 
        formalism with noise-weighted inner products between gravitational wave signals and detector 
        noise power spectral densities. It supports multiprocessing for efficient computation and 
        is compatible with various waveform approximants from LALSimulation.

        The method generates frequency-domain waveforms using LAL, computes the inner products 
        with detector PSDs, and calculates antenna response patterns for each detector in the 
        network. It automatically handles duration estimation based on chirp time and supports 
        systems with arbitrary spin configurations including precession.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar masses. Default is 10.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar masses. Default is 10.
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Gravitational wave polarization angle in radians. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Gravitational wave phase at coalescence in radians. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the source in radians. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the source in radians. Default is 0.0.
        a_1 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the primary object. Default is 0.0.
        a_2 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the secondary object. Default is 0.0.
        tilt_1 : `numpy.ndarray` or `float`
            Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.
        tilt_2 : `numpy.ndarray` or `float`
            Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.
        phi_12 : `numpy.ndarray` or `float`
            Azimuthal angle between the two spins in radians. Default is 0.0.
        phi_jl : `numpy.ndarray` or `float`
            Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.
        lambda_1 : `numpy.ndarray` or `float`
            Dimensionless tidal deformability of primary object. Default is 0.0.
        lambda_2 : `numpy.ndarray` or `float`
            Dimensionless tidal deformability of secondary object. Default is 0.0.
        eccentricity : `numpy.ndarray` or `float`
            Orbital eccentricity at reference frequency. Default is 0.0.
        gw_param_dict : `dict` or `bool`
            Dictionary containing all gravitational wave parameters as key-value pairs. 
            If provided, takes precedence over individual parameter arguments. Default is False.
        output_jsonfile : `str` or `bool`
            If string, saves the SNR results to a JSON file with the given filename. 
            If True, saves to 'snr.json'. If False, no file output. Default is False.

        Returns
        -------
        optimal_snr : `dict`
            Dictionary containing SNR values for each detector and network SNR.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'optimal_snr_net'.
            Values are numpy arrays of SNR values corresponding to input parameters.
            Systems with total mass outside [mtot_min, mtot_max] have SNR set to zero.

        Notes
        -----
        - Uses LALSimulation for frequency-domain waveform generation
        - Automatically estimates waveform duration based on chirp time with safety factor
        - Duration is bounded by :attr:`~duration_min` and :attr:`~duration_max` if specified
        - Supports multiprocessing with :attr:`~npool` processors for parallel computation
        - Compatible with all LAL waveform approximants including precessing and higher-order modes
        - Uses :func:`~gwsnr.utils.noise_weighted_inner_prod` for inner product calculation
        - Antenna response patterns computed using :func:`~gwsnr.numba.antenna_response_array`

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> # Initialize with inner product method
        >>> snr = GWSNR(snr_type='inner_product')
        >>> # Calculate SNR for aligned systems
        >>> result = snr.compute_bilby_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
        
        >>> # Calculate SNR for precessing systems
        >>> result = snr.compute_bilby_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0,
        ...                               a_1=0.5, a_2=0.3, tilt_1=0.2, tilt_2=0.1)
        
        >>> # Using parameter dictionary
        >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
        >>> result = snr.compute_bilby_snr(gw_param_dict=params)
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

        # to access multi-cores instead of multithreading
        if mp.current_process().name != 'MainProcess':
            print(
                "\n\n[ERROR] This multiprocessing code must be run under 'if __name__ == \"__main__\":'.\n"
                "Please wrap your script entry point in this guard.\n"
                "See: https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming\n"
            )
            raise RuntimeError(
                "\nMultiprocessing code must be run under 'if __name__ == \"__main__\":'.\n\n"
            )

        with mp.Pool(processes=npool) as pool:
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
        
        # close forked processes
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
        Function to calculate SNR using JAX-accelerated noise-weighted inner product method with Ripple waveform generation.

        This method computes the optimal signal-to-noise ratio using JAX-accelerated inner products between 
        gravitational wave signals generated with the Ripple waveform generator and detector noise power 
        spectral densities. It leverages JAX's just-in-time (JIT) compilation and vectorized map (vmap) 
        functions for highly efficient parallelized computation, making it suitable for large-scale 
        parameter estimation and population studies.

        The method uses the RippleInnerProduct class for waveform generation and inner product calculation,
        automatically handling duration estimation and supporting arbitrary spin configurations. It provides
        significant computational speedup compared to traditional LAL-based methods while maintaining
        numerical accuracy.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar masses. Default is 10.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar masses. Default is 10.
        luminosity_distance : `numpy.ndarray` or `float`
            Luminosity distance of the binary in Mpc. Default is 100.0.
        theta_jn : `numpy.ndarray` or `float`
            Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.
        psi : `numpy.ndarray` or `float`
            Gravitational wave polarization angle in radians. Default is 0.0.
        phase : `numpy.ndarray` or `float`
            Gravitational wave phase at coalescence in radians. Default is 0.0.
        geocent_time : `numpy.ndarray` or `float`
            GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.
        ra : `numpy.ndarray` or `float`
            Right ascension of the source in radians. Default is 0.0.
        dec : `numpy.ndarray` or `float`
            Declination of the source in radians. Default is 0.0.
        a_1 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the primary object. Default is 0.0.
        a_2 : `numpy.ndarray` or `float`
            Dimensionless spin magnitude of the secondary object. Default is 0.0.
        tilt_1 : `numpy.ndarray` or `float`
            Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.
        tilt_2 : `numpy.ndarray` or `float`
            Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.
        phi_12 : `numpy.ndarray` or `float`
            Azimuthal angle between the two spins in radians. Default is 0.0.
        phi_jl : `numpy.ndarray` or `float`
            Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.
        gw_param_dict : `dict` or `bool`
            Dictionary containing all gravitational wave parameters as key-value pairs. 
            If provided, takes precedence over individual parameter arguments. Default is False.
        output_jsonfile : `str` or `bool`
            If string, saves the SNR results to a JSON file with the given filename. 
            If True, saves to 'snr.json'. If False, no file output. Default is False.

        Returns
        -------
        optimal_snr : `dict`
            Dictionary containing SNR values for each detector and network SNR.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'optimal_snr_net'.
            Values are numpy arrays of SNR values corresponding to input parameters.
            Systems with total mass outside [mtot_min, mtot_max] have SNR set to zero.

        Notes
        -----
        - Uses Ripple waveform generator with JAX backend for GPU acceleration
        - Automatically estimates waveform duration bounded by :attr:`~duration_min` and :attr:`~duration_max`
        - Compatible with waveform approximants supported by Ripple (e.g., IMRPhenomD, IMRPhenomXPHM)
        - Leverages JAX's jit and vmap for vectorized batch processing
        - Supports multiprocessing with :attr:`~npool` processors when applicable
        - Uses :meth:`~RippleInnerProduct.noise_weighted_inner_product_jax` for inner product calculation
        - Antenna response patterns computed using :func:`~gwsnr.numba.antenna_response_array`
        - Requires :attr:`~snr_type` to be set to 'inner_product_jax' during GWSNR initialization

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> # Initialize with JAX inner product method
        >>> snr = GWSNR(snr_type='inner_product_jax', waveform_approximant='IMRPhenomD')
        >>> # Calculate SNR for aligned systems
        >>> result = snr.compute_ripple_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
        
        >>> # Calculate SNR for precessing systems
        >>> result = snr.compute_ripple_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0,
        ...                                a_1=0.5, a_2=0.3, tilt_1=0.2, tilt_2=0.1)
        
        >>> # Using parameter dictionary
        >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
        >>> result = snr.compute_ripple_snr(gw_param_dict=params)
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
        Function to calculate probability of detection for gravitational wave signals using SNR threshold criteria.

        This method computes the probability of detection (Pdet) for gravitational wave signals based on 
        signal-to-noise ratio thresholds for individual detectors and the detector network. It supports 
        both matched filter probability calculation using Gaussian noise assumptions and simple boolean 
        threshold detection. The method is compatible with single or multiple SNR threshold values for 
        different detectors in the network.

        Parameters
        ----------
        snr_dict : `dict`
            Dictionary containing SNR values for each detector and network SNR.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'optimal_snr_net'.
            Values are numpy arrays of SNR values corresponding to input parameters.
        snr_th : `float` or `numpy.ndarray` or `None`
            SNR threshold for individual detector detection. If None, uses :attr:`~snr_th`.
            If array, must have length equal to number of detectors. Default is None.
        snr_th_net : `float` or `None`
            SNR threshold for network detection. If None, uses :attr:`~snr_th_net`. Default is None.
        type : `str`
            Type of probability calculation method. Default is 'matched_filter'.
            Options: 'matched_filter' (Gaussian noise probability), 'bool' (boolean threshold).

        Returns
        -------
        pdet_dict : `dict`
            Dictionary containing probability of detection for each detector and network.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'pdet_net'.
            Values are numpy arrays of detection probabilities [0,1] for 'matched_filter' 
            or boolean arrays {0,1} for 'bool' type.

        Notes
        -----
        - For 'matched_filter' type: Uses Gaussian noise assumption with Pdet = 1 - Φ(ρ_th - ρ)
          where Φ is the cumulative distribution function of the standard normal distribution
        - For 'bool' type: Returns 1 if SNR > threshold, 0 otherwise
        - Individual detector thresholds can be different by providing array of thresholds
        - Network detection uses quadrature sum of individual detector SNRs
        - Compatible with all SNR calculation methods (interpolation, inner product, ANN)

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> snr = GWSNR(snr_type='interpolation', pdet=True)
        >>> # Calculate SNR first
        >>> snr_result = snr.snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
        >>> # Calculate detection probability manually
        >>> pdet_result = snr.probability_of_detection(snr_result, snr_th=8.0, type='matched_filter')
        
        >>> # Using different thresholds for different detectors
        >>> pdet_result = snr.probability_of_detection(snr_result, snr_th=[8.0, 8.0, 7.0], type='bool')
        """

        if snr_th:
            snr_th = snr_th
        else:
            snr_th = self.snr_th

        # check if snr_th is an array or a single value
        if isinstance(snr_th, (list, np.ndarray)):
            snr_th = np.array(snr_th)
        else:
            snr_th = np.full(len(self.detector_list), snr_th)

        if snr_th_net:
            snr_th_net = snr_th_net
        else:
            snr_th_net = self.snr_th_net

        detectors = np.array(self.detector_list)
        pdet_dict = {}
        for i, det in enumerate(detectors):
            if type == "matched_filter":
                pdet_dict[det] = np.array(1 - norm.cdf(snr_th[i] - snr_dict[det]))
            else:
                pdet_dict[det] = np.array(snr_th[i] < snr_dict[det], dtype=int)

        if type == "matched_filter":
            pdet_dict["pdet_net"] = np.array(1 - norm.cdf(snr_th_net - snr_dict["optimal_snr_net"]))
        else:
            pdet_dict["pdet_net"] = np.array(snr_th_net < snr_dict["optimal_snr_net"], dtype=int)

        return pdet_dict

    def detector_horizon(self, mass_1=1.4, mass_2=1.4, snr_th=None, snr_th_net=None):
        """
        Function to calculate detector horizon distance for compact binary coalescences.

        This method computes the horizon distance for each detector in the network, defined as the 
        luminosity distance at which a compact binary coalescence would produce a signal-to-noise 
        ratio equal to the detection threshold. The horizon distance represents the maximum range 
        at which a source can be detected with optimal orientation and sky location.

        The calculation uses a reference binary system (typically BNS with masses m1=m2=1.4 M☉) 
        at optimal orientation (face-on, overhead) and scales the SNR to find the distance where 
        the SNR equals the detection threshold. The method accounts for detector antenna response 
        patterns and uses the same waveform generation as other SNR calculation methods.

        Parameters
        ----------
        mass_1 : `numpy.ndarray` or `float`
            Primary mass of the binary in solar masses. Default is 1.4.
        mass_2 : `numpy.ndarray` or `float`
            Secondary mass of the binary in solar masses. Default is 1.4.
        snr_th : `float` or `None`
            SNR threshold for individual detector detection. If None, uses :attr:`~snr_th`. Default is None.
        snr_th_net : `float` or `None`
            SNR threshold for network detection. If None, uses :attr:`~snr_th_net`. Default is None.

        Returns
        -------
        horizon : `dict`
            Dictionary containing horizon distances for each detector and network.
            Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'net'.
            Values are horizon distances in Mpc for the given binary system and SNR thresholds.

        Notes
        -----
        - Uses optimal orientation: theta_jn=0 (face-on), ra=dec=psi=phase=0 (overhead)
        - Reference luminosity distance is 100 Mpc for SNR calculation scaling
        - Horizon distance = (d_eff/SNR_th) × SNR_100Mpc where d_eff is effective distance
        - Network horizon uses quadrature sum of effective distances from all detectors
        - Compatible with all waveform approximants supported by the inner product method
        - Uses :meth:`~compute_bilby_snr` for reference SNR calculation at 100 Mpc

        Examples
        --------
        >>> from gwsnr import GWSNR
        >>> snr = GWSNR(snr_type='inner_product')
        >>> # Calculate BNS horizon for default 1.4+1.4 solar mass system
        >>> horizon = snr.detector_horizon()
        >>> print(f"LIGO-Hanford horizon: {horizon['H1']:.1f} Mpc")
        
        >>> # Calculate horizon for different mass system
        >>> horizon_bbh = snr.detector_horizon(mass_1=30.0, mass_2=30.0, snr_th=8.0)
        >>> print(f"Network horizon: {horizon_bbh['net']:.1f} Mpc")
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
                antenna_response_plus(ra_, dec_, geocent_time_, psi_, tensor)
                for tensor in detector_tensor
            ]
        )
        Fc = np.array(
            [
                antenna_response_cross(ra_, dec_, geocent_time_, psi_, tensor)
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
    

# def set_multiprocessing_start_method():
#     import sys
    
#     if mp.current_process().name == "MainProcess":
#         if not hasattr(sys, 'ps1'):
#             # Not running in main script or interactive mode (Jupyter/IPython)
        
#             # Option 1: Print warning (less disruptive)
#             print("\n[ERROR] This multiprocessing code must be run under 'if __name__ == \"__main__\":'.\n"
#                 "Please wrap your script entry point in this guard.\n"
#                 "See: https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming\n")
#             # Option 2: Raise Exception (fail fast, preferred for libraries)
#             raise RuntimeError(
#                 "\n\nMultiprocessing code must be run under 'if __name__ == \"__main__\":'.\n"
#                 "Please wrap your script entry point in this guard.\n"
#                 "See: https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming\n\n")