# -*- coding: utf-8 -*-
"""
This module provides efficient calculation of signal-to-noise ratio (SNR) and probability of detection (Pdet) 
for gravitational wave signals from compact binary coalescences.

The module supports multiple computational backends including interpolation-based methods for fast calculation,
inner product methods with LAL waveforms, JAX-accelerated computation, and artificial neural networks.
It handles various detector configurations, waveform approximants, and spin scenarios.

Key Features:
- Fast SNR calculation via bicubic interpolation of precomputed coefficients
- Inner product methods with LAL and Ripple waveform generators  
- JAX and MLX acceleration for GPU/vectorized computation
- ANN-based detection probability estimation
- Multi-detector network analysis with antenna response patterns
- Support for aligned and precessing spin systems
- Probability of detection calculations with various statistical models
- Detector horizon distance estimation: analytical and numerical methods

Copyright (C) 2025 Hemantakumar Phurailatpam and Otto Hannuksela. 
Distributed under MIT License.
"""

import shutil
import os
from importlib.resources import path
import pathlib

import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from scipy.stats import norm, ncx2

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
from ..utils import noise_weighted_inner_prod_h_inner_h, noise_weighted_inner_prod_d_inner_h  # from gwsnr/utils/multiprocessing_routine.py

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
    Calculate signal-to-noise ratio (SNR) and detection probability for gravitational wave signals.

    This class provides multiple computational methods for SNR calculation:
    - Fast interpolation using precomputed coefficients
    - Noise-weighted inner products with LAL/Ripple waveforms
    - JAX/MLX acceleration for GPU computation
    - Neural network estimation for population studies

    Supports various detector networks, waveform approximants, and spin configurations.

    Parameters
    ----------
    npool : int, default=4
        Number of processors for parallel processing.
    mtot_min : float, default=9.96=2*4.98
        Minimum total mass in solar masses for interpolation grid. 4.98 Mo is the minimum component mass of BBH systems in GWTC-3. 9.96=2*4.98
    mtot_max : float, default=235.0=2*112.5+10.0
        Maximum total mass in solar masses. Auto-adjusted if mtot_cut=True. 112.5 Mo is the maximum component mass of BBH systems in GWTC-3. 10.0 Mo is added to avoid edge effects.
    ratio_min : float, default=0.1
        Minimum mass ratio (m2/m1) for interpolation grid.
    ratio_max : float, default=1.0
        Maximum mass ratio for interpolation grid.
    spin_max : float, default=0.99
        Maximum aligned spin magnitude for interpolation methods.
    mtot_resolution : int, default=200
        Number of total mass grid points for interpolation.
    ratio_resolution : int, default=20
        Number of mass ratio grid points for interpolation.
    spin_resolution : int, default=10
        Number of spin grid points for aligned-spin methods.
    batch_size_interpolation : int, default=1000000
        Batch size for interpolation calculations.
    sampling_frequency : float, default=2048.0
        Detector sampling frequency in Hz.
    waveform_approximant : str, default='IMRPhenomD'
        Waveform model (e.g., 'IMRPhenomD', 'IMRPhenomXPHM', 'TaylorF2').
    frequency_domain_source_model : str, default='lal_binary_black_hole'
        LAL source model for waveform generation.
    minimum_frequency : float, default=20.0
        Minimum frequency in Hz for waveform generation.
    reference_frequency : float, optional
        Reference frequency in Hz. Defaults to minimum_frequency.
    duration_max : float, optional
        Maximum waveform duration in seconds. Auto-set for some approximants.
    duration_min : float, optional
        Minimum waveform duration in seconds.
    fixed_duration : float, optional
        Fixed duration for all waveforms if specified.
    snr_method : str, default='interpolation_no_spins'
        SNR calculation method:
        - 'interpolation_no_spins[_jax/_mlx]': Fast interpolation without spins
        - 'interpolation_aligned_spins[_jax/_mlx]': With aligned spins
        - 'inner_product[_jax]': Direct inner product calculation
        - 'ann': Artificial neural network estimation
    snr_type : str, default='optimal_snr'
        Type of SNR ('optimal_snr' or 'observed_snr').
    noise_realization : array_like, optional
        Noise realization for observed SNR (not yet implemented).
    psds : dict, optional
        Power spectral densities for detectors. Options:
        - None: Use bilby defaults
        - {'H1': 'aLIGODesign', 'L1': 'aLIGODesign'}: PSD names
        - {'H1': 'custom_psd.txt'}: Custom PSD files
        - {'H1': 1234567890}: GPS time for data-based PSD
    ifos : list, optional
        Custom interferometer objects. None uses defaults from psds.
    interpolator_dir : str, default='./interpolator_pickle'
        Directory for interpolation coefficient storage.
    create_new_interpolator : bool, default=False
        Force generation of new interpolation coefficients.
    gwsnr_verbose : bool, default=True
        Print initialization parameters.
    multiprocessing_verbose : bool, default=True
        Show progress bars during computation.
    mtot_cut : bool, default=False
        Limit mtot_max based on minimum_frequency to avoid undetectable systems.
    pdet_kwargs : dict, optional
        Detection probability parameters:
        - 'snr_th': Individual detector threshold (default=8.0)
        - 'snr_th_net': Network threshold (default=8.0)  
        - 'pdet_type': 'boolean' or 'probability_distribution'
        - 'distribution_type': 'gaussian' or 'noncentral_chi2'
    ann_path_dict : dict or str, optional
        ANN model paths. None uses built-in models.
    snr_recalculation : bool, default=False
        Enable hybrid recalculation near detection threshold.
    snr_recalculation_range : list, default=[4,12]
        SNR range for triggering recalculation.
    snr_recalculation_waveform_approximant : str, default='IMRPhenomXPHM'
        Waveform for recalculation.

    Examples
    --------
    Basic usage with interpolation:
    
    >>> from gwsnr import GWSNR
    >>> snr_calc = GWSNR(snr_method='interpolation_no_spins')
    >>> result = snr_calcoptimal_snr(mass_1=30, mass_2=25, luminosity_distance=100)
    >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")

    With aligned spins:
    
    >>> snr_calc = GWSNR(snr_method='interpolation_aligned_spins')
    >>> result = snr_calcoptimal_snr(mass_1=30, mass_2=25, a_1=0.5, a_2=-0.3)

    Detection probability:
    
    >>> pdet_calc = GWSNR(pdet_kwargs={'snr_th': 8})
    >>> pdet = pdet_calc.pdet(mass_1=30, mass_2=25, luminosity_distance=200)

    Notes
    -----
    - Interpolation methods are fastest for population studies
    - Inner product methods are most accurate for individual events  
    - JAX methods leverage GPU acceleration when available
    - ANN methods provide fast detection probability estimates, but less accurate SNRs
    """

    # Class attributes with documentation
    npool = None
    """``int`` \n
    Number of processors for parallel processing."""

    mtot_min = None
    """``float`` \n
    Minimum total mass (M☉) for interpolation grid."""

    mtot_max = None
    """``float`` \n
    Maximum total mass (M☉) for interpolation grid."""

    ratio_min = None
    """``float`` \n
    Minimum mass ratio (q = m2/m1) for interpolation grid."""

    ratio_max = None
    """``float`` \n
    Maximum mass ratio for interpolation grid."""

    spin_max = None
    """``float`` \n
    Maximum aligned spin magnitude for interpolation."""

    mtot_resolution = None
    """``int`` \n
    Grid resolution for total mass interpolation."""

    ratio_resolution = None
    """``int`` \n
    Grid resolution for mass ratio interpolation."""

    spin_resolution = None
    """``int`` \n
    Grid resolution for aligned spin interpolation."""

    ratio_arr = None
    """``numpy.ndarray`` \n
    Mass ratio interpolation grid points."""

    mtot_arr = None
    """``numpy.ndarray`` \n
    Total mass interpolation grid points."""

    a_1_arr = None
    """``numpy.ndarray`` \n
    Primary aligned spin interpolation grid."""

    a_2_arr = None
    """``numpy.ndarray`` \n
    Secondary aligned spin interpolation grid."""

    sampling_frequency = None
    """``float`` \n
    Detector sampling frequency (Hz)."""

    waveform_approximant = None
    """``str`` \n
    LAL waveform approximant (e.g., 'IMRPhenomD', 'IMRPhenomXPHM')."""

    frequency_domain_source_model = None
    """``str`` \n
    LAL frequency domain source model."""

    f_min = None
    """``float`` \n
    Minimum waveform frequency (Hz)."""

    f_ref = None
    """``float`` \n
    Reference frequency (Hz) for waveform generation."""

    duration_max = None
    """``float`` or ``None`` \n
    Maximum waveform duration (s). Auto-set if None."""

    duration_min = None
    """``float`` or ``None`` \n
    Minimum waveform duration (s). Auto-set if None."""

    snr_method = None
    """``str`` \n
    SNR calculation method. Options: interpolation variants, inner_product variants, ann."""

    snr_type = None
    """``str`` \n
    SNR type: 'optimal_snr' or 'observed_snr' (not implemented)."""

    noise_realization = None
    """``numpy.ndarray`` or ``None`` \n
    Noise realization for observed SNR (not implemented)."""

    psds_list = None
    """``list`` of ``PowerSpectralDensity`` \n
    Detector power spectral densities."""

    detector_tensor_list = None
    """``list`` of ``numpy.ndarray`` \n
    Detector tensors for antenna response calculations."""

    detector_list = None
    """``list`` of ``str`` \n
    Detector names (e.g., ['H1', 'L1', 'V1'])."""

    ifos = None
    """``list`` of ``Interferometer`` \n
    Bilby interferometer objects."""

    interpolator_dir = None
    """``str`` \n
    Directory for interpolation coefficient storage."""

    path_interpolator = None
    """``list`` of ``str`` \n
    Paths to interpolation coefficient files."""

    snr_partialsacaled_list = None
    """``list`` of ``numpy.ndarray`` \n
    Partial-scaled SNR interpolation coefficients."""

    multiprocessing_verbose = None
    """``bool`` \n
    Show progress bars for multiprocessing computations."""

    param_dict_given = None
    """``dict`` \n
    Interpolator parameter dictionary for caching."""

    snr_th = None
    """``float`` \n
    Individual detector SNR threshold (default: 8.0)."""

    snr_th_net = None
    """``float`` \n
    Network SNR threshold (default: 8.0)."""

    model_dict = None
    """``dict`` \n
    ANN models for each detector (when snr_method='ann')."""

    scaler_dict = None
    """``dict`` \n
    ANN feature scalers for each detector (when snr_method='ann')."""

    error_adjustment = None
    """``dict`` \n
    ANN error correction parameters (when snr_method='ann')."""

    ann_catalogue = None
    """``dict`` \n
    ANN model configuration and paths (when snr_method='ann')."""

    snr_recalculation = None
    """``bool`` \n
    Enable hybrid SNR recalculation near detection threshold."""

    snr_recalculation_range = None
    """``list`` \n
    SNR range [min, max] triggering recalculation."""

    snr_recalculation_waveform_approximant = None
    """``str`` \n
    Waveform approximant for SNR recalculation."""

    get_interpolated_snr = None
    """``function`` \n
    Interpolated SNR calculation function (backend-specific)."""

    noise_weighted_inner_product_jax = None
    """``function`` \n
    JAX-accelerated inner product function (when snr_method='inner_product_jax')."""

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
        batch_size_interpolation=1000000,
        sampling_frequency=2048.0,
        waveform_approximant="IMRPhenomD",
        frequency_domain_source_model='lal_binary_black_hole',
        minimum_frequency=20.0,
        reference_frequency=None,
        duration_max=None,
        duration_min=None,
        fixed_duration=None,
        snr_method="interpolation_no_spins",
        snr_type="optimal_snr",
        noise_realization=None,
        psds=None,
        ifos=None,
        interpolator_dir="./interpolator_pickle",
        create_new_interpolator=False,
        gwsnr_verbose=True,
        multiprocessing_verbose=True,
        mtot_cut=False,
        pdet_kwargs=None,
        ann_path_dict=None,
        snr_recalculation=False,
        snr_recalculation_range=[4,12],
        snr_recalculation_waveform_approximant="IMRPhenomXPHM",
    ):

        print("\nInitializing GWSNR class...\n")
        # setting instance attributes
        self.npool = npool
        self.pdet_kwargs = pdet_kwargs if pdet_kwargs is not None else dict(snr_th=8.0, snr_th_net=8.0, pdet_type='boolean', distribution_type='noncentral_chi2')
        self.duration_max = duration_max
        self.duration_min = duration_min
        self.fixed_duration = fixed_duration
        self.snr_method = snr_method
        self.snr_type = snr_type

        if self.snr_method=='observed_snr':
            raise ValueError("'observed_snr' not implemented yet. Use 'optimal_snr' instead.")
        
        self.noise_realization = noise_realization
        self.spin_max = spin_max
        self.batch_size_interpolation = batch_size_interpolation

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
        self.f_ref = reference_frequency
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
        psds_list, detector_tensor_list, detector_list, self.ifos  = dealing_with_psds(
            psds, ifos, minimum_frequency, sampling_frequency
        )

        # param_dict_given is an identifier for the interpolator
        self.param_dict_given = {
            "mtot_min": self.mtot_min,
            "mtot_max": self.mtot_max,
            "mtot_resolution": self.mtot_resolution,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "spin_max": self.spin_max,
            "ratio_resolution": self.ratio_resolution,
            "sampling_frequency": self.sampling_frequency,
            "waveform_approximant": self.waveform_approximant,
            "minimum_frequency": self.f_min,
            "reference_frequency": self.f_ref if self.f_ref is not None else self.f_min,
            "duration_max": self.duration_max,
            "duration_min": self.duration_min,
            "fixed_duration": self.fixed_duration,
            "frequency_domain_source_model": self.frequency_domain_source_model,
            "detector": detector_list,
            "psds": psds_list,
            "detector_tensor": detector_tensor_list,
        }
        if waveform_approximant=="IMRPhenomXPHM" and duration_max is None:
            print("Intel processor has trouble allocating memory when the data is huge. So, by default for IMRPhenomXPHM, duration_max = 64.0. Otherwise, set to some max value like duration_max = 600.0 (10 mins)")
            self.duration_max = 64.0
            self.duration_min = 4.0


        # now generate interpolator, if not exists
        list_no_spins = ["interpolation", "interpolation_no_spins", "interpolation_no_spins_numba", "interpolation_no_spins_jax", "interpolation_no_spins_mlx"]
        list_aligned_spins = ["interpolation_aligned_spins", "interpolation_aligned_spins_numba", "interpolation_aligned_spins_jax", "interpolation_aligned_spins_mlx"]

        if snr_method in list_no_spins:

            if snr_method == "interpolation_no_spins_jax":
                from ..jax import get_interpolated_snr_no_spins_jax
                self.get_interpolated_snr = get_interpolated_snr_no_spins_jax
            elif snr_method == "interpolation_no_spins_mlx":
                from ..mlx import get_interpolated_snr_no_spins_mlx
                self.get_interpolated_snr = get_interpolated_snr_no_spins_mlx
            else:
                from ..numba import get_interpolated_snr_no_spins_numba
                self.get_interpolated_snr = get_interpolated_snr_no_spins_numba

            # dealing with interpolator
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

        elif snr_method in list_aligned_spins:

            if snr_method == "interpolation_aligned_spins_jax":
                from ..jax import get_interpolated_snr_aligned_spins_jax
                self.get_interpolated_snr = get_interpolated_snr_aligned_spins_jax
            elif snr_method == "interpolation_aligned_spins_mlx":
                from ..mlx import get_interpolated_snr_aligned_spins_mlx
                self.get_interpolated_snr = get_interpolated_snr_aligned_spins_mlx
            else:
                from ..numba import get_interpolated_snr_aligned_spins_numba
                self.get_interpolated_snr = get_interpolated_snr_aligned_spins_numba

            self.param_dict_given['spin_max'] = self.spin_max
            self.param_dict_given['spin_resolution'] = self.spin_resolution
            # dealing with interpolator
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

        # inner product method doesn't need interpolator generation
        elif snr_method == "inner_product":
            pass
        
        # need to initialize RippleInnerProduct class
        elif snr_method == "inner_product_jax":
            from ..ripple import RippleInnerProduct

            ripple_class = RippleInnerProduct(
                waveform_name=waveform_approximant, 
                minimum_frequency=self.f_min, 
                sampling_frequency=sampling_frequency, 
                reference_frequency=self.f_ref if self.f_ref is not None else self.f_min,
                )

            self.noise_weighted_inner_product_jax = ripple_class.noise_weighted_inner_product_jax

        # ANN method still needs the partialscaledSNR interpolator.
        elif snr_method == "ann":

            from ..numba import get_interpolated_snr_aligned_spins_numba
            self.get_interpolated_snr = get_interpolated_snr_aligned_spins_numba
            # below is added to find the genereated interpolator path
            self.param_dict_given['spin_max'] = self.spin_max
            self.param_dict_given['spin_resolution'] = self.spin_resolution
            
            self.model_dict, self.scaler_dict, self.error_adjustment, self.ann_catalogue = self.ann_initilization(ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant, snr_th)
            # dealing with interpolator
            self.snr_method = "interpolation_aligned_spins"
            self.path_interpolator = self.interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)
            self.snr_method = "ann"

        else:
            raise ValueError("SNR function type not recognised. Please choose from 'interpolation', 'interpolation_no_spins', 'interpolation_no_spins_numba', 'interpolation_no_spins_jax', 'interpolation_no_spins_mlx', 'interpolation_aligned_spins', 'interpolation_aligned_spins_numba', 'interpolation_aligned_spins_jax', 'interpolation_aligned_spins_mlx', 'inner_product', 'inner_product_jax', 'ann'.")

        # change back to original
        self.psds_list = psds_list
        self.detector_tensor_list = detector_tensor_list
        self.detector_list = detector_list

        if (snr_method == "inner_product") or (snr_method == "inner_product_jax"):
            self.optimal_snr_with_interpolation = self._print_no_interpolator

        # print some info
        self.print_all_params(gwsnr_verbose)
        print("\n")

    # dealing with interpolator
    def interpolator_setup(self, interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list):
        """
        Set up interpolator files for fast SNR calculation using precomputed coefficients.

        This method manages the creation and loading of partialscaled SNR interpolation data.
        It checks for existing interpolators, generates missing ones, and loads coefficients
        for runtime use.

        Parameters
        ----------
        interpolator_dir : str
            Directory path for storing interpolator pickle files.
        create_new_interpolator : bool  
            If True, generates new interpolators regardless of existing files.
        psds_list : list
            Power spectral density objects for each detector.
        detector_tensor_list : list
            Detector tensor arrays for antenna response calculations.
        detector_list : list
            Detector names (e.g., ['L1', 'H1', 'V1']).

        Returns
        -------
        path_interpolator_all : list
            File paths to interpolator pickle files for all detectors.

        Notes
        -----
        - Uses :func:`interpolator_check` to identify missing interpolators
        - Calls :meth:`init_partialscaled` to generate new coefficients
        - Loads coefficients into :attr:`snr_partialsacaled_list` for runtime use
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
            # if self.snr_method == 'interpolation_aligned_spins':
            #     self.init_partialscaled_aligned_spins()
            # else:
            self.init_partialscaled()
        elif create_new_interpolator:
            # change back to original
            self.psds_list = psds_list
            self.detector_tensor_list = detector_tensor_list
            self.detector_list = detector_list
            print("Please be patient while the interpolator is generated")
            # if self.snr_method == 'interpolation_aligned_spins':
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
        Initialize ANN models and feature scalers for detection probability estimation.

        Loads pre-trained neural network models, feature scalers, and error correction parameters
        for each detector. Validates that model parameters match current GWSNR configuration.

        Parameters
        ----------
        ann_path_dict : dict, str, or None
            Dictionary or JSON file path containing ANN model paths for each detector.
            If None, uses default models from gwsnr/ann/data/ann_path_dict.json.
            Expected structure: {detector_name: {'model_path': str, 'scaler_path': str, 
            'error_adjustment_path': str, 'sampling_frequency': float, 'minimum_frequency': float, 
            'waveform_approximant': str, 'snr_th': float}}.
        detector_list : list of str
            Detector names requiring ANN models (e.g., ['L1', 'H1', 'V1']).
        sampling_frequency : float
            Sampling frequency in Hz. Must match ANN training configuration.
        minimum_frequency : float
            Minimum frequency in Hz. Must match ANN training configuration.
        waveform_approximant : str
            Waveform model. Must match ANN training configuration.
        snr_th : float
            Detection threshold. Must match ANN training configuration.

        Returns
        -------
        model_dict : dict
            Loaded TensorFlow/Keras models {detector_name: model}.
        scaler_dict : dict
            Feature preprocessing scalers {detector_name: scaler}.
        error_adjustment : dict
            Post-prediction correction parameters {detector_name: {'slope': float, 'intercept': float}}.
        ann_catalogue : dict
            Complete ANN configuration and paths for all detectors.

        Raises
        ------
        ValueError
            If model not available for detector, or if model parameters don't match 
            current GWSNR configuration.

        Notes
        -----
        - Loads models from gwsnr/ann/data if file paths don't exist locally
        - Validates parameter compatibility before loading
        - Error adjustment improves prediction accuracy via linear correction
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
        Print error message when interpolation methods are called without available interpolators.

        This placeholder method is assigned to :meth:`~optimal_snr_with_interpolation` when using 
        inner product SNR methods that don't require interpolation coefficients.

        Parameters
        ----------
        **kwargs : `dict`
            Arbitrary keyword arguments (ignored).

        Raises
        ------
        ValueError
            Always raised, suggesting to use interpolation-based :attr:`~snr_method`.
        """

        raise ValueError(
            'No interpolator found. Please set snr_method="interpolation" to generate new interpolator.'
        )

    def calculate_mtot_max(self, mtot_max, minimum_frequency):
        """
        Calculate maximum total mass cutoff based on minimum frequency to ensure positive chirp time.

        This method finds the maximum total mass where the chirp time becomes zero at the given 
        minimum frequency. Systems with higher masses would have negative chirp times, causing 
        waveform generation failures. A safety factor of 1.1 is applied.

        Parameters
        ----------
        mtot_max : float
            User-specified maximum total mass in solar masses.
        minimum_frequency : float
            Minimum frequency in Hz for waveform generation.

        Returns
        -------
        float
            Adjusted maximum total mass (≤ input mtot_max) ensuring positive chirp time.

        Notes
        -----
        Uses equal mass ratio (q=1.0) as conservative estimate since it maximizes chirp time
        for given total mass. Particularly important for TaylorF2 approximant.
        """

        # Note: mass ratio is fixed at 1.0 because it gives the highest chirp time for a given mtot
        def func(x, mass_ratio=1.0):
            mass_1 = x / (1 + mass_ratio)
            mass_2 = x / (1 + mass_ratio) * mass_ratio

            return 1.1*findchirp_chirptime(mass_1, mass_2, minimum_frequency)

        # find where func is zero
        from scipy.optimize import fsolve
        
        mtot_max_generated = fsolve(func, 184)[
            0
        ]  # to make sure that chirptime is not negative, TaylorF2 might need this
        if mtot_max > mtot_max_generated:
            mtot_max = mtot_max_generated

        return mtot_max

    def print_all_params(self, verbose=True):
        """
        Print all parameters and configuration of the GWSNR class instance.

        Displays computational settings, waveform configuration, detector setup, mass parameter 
        ranges, and interpolation parameters for verification and debugging.

        Parameters
        ----------
        verbose : bool, default=True
            If True, print all parameters to stdout. If False, suppress output.

        Notes
        -----
        Printed information includes:
        - Computational: processors, SNR method
        - Waveform: approximant, frequencies, sampling rate  
        - Detectors: names and PSDs
        - Mass ranges: total mass bounds with frequency cutoffs
        - Interpolation: grid resolutions and bounds (when applicable)

        Called automatically during initialization when gwsnr_verbose=True.
        """

        if verbose:
            print("\nChosen GWSNR initialization parameters:\n")
            print("npool: ", self.npool)
            print("snr type: ", self.snr_method)
            print("waveform approximant: ", self.waveform_approximant)
            print("sampling frequency: ", self.sampling_frequency)
            print("minimum frequency (fmin): ", self.f_min)
            print("reference frequency (f_ref): ", self.f_ref if self.f_ref is not None else self.f_min)
            print("mtot=mass1+mass2")
            print("min(mtot): ", self.mtot_min)
            print(
                f"max(mtot) (with the given fmin={self.f_min}): {self.mtot_max}",
            )
            print("detectors: ", self.detector_list)
            print("psds: ", self.psds_list)
            if self.snr_method == "interpolation":
                print("min(ratio): ", self.ratio_min)
                print("max(ratio): ", self.ratio_max)
                print("mtot resolution: ", self.mtot_resolution)
                print("ratio resolution: ", self.ratio_resolution)
                print("interpolator directory: ", self.interpolator_dir)

    def optimal_snr(
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
        Calculate optimal SNR for gravitational wave signals from compact binary coalescences.

        This is the primary interface for SNR calculation, routing to the appropriate computational method
        based on the configured snr_method. Supports interpolation, inner product, JAX-accelerated, and
        neural network methods.

        Parameters
        ----------
        mass_1 : array_like or float, default=np.array([10.0])
            Primary mass in solar masses.
        mass_2 : array_like or float, default=np.array([10.0])  
            Secondary mass in solar masses.
        luminosity_distance : array_like or float, default=100.0
            Luminosity distance in Mpc.
        theta_jn : array_like or float, default=0.0
            Inclination angle (total angular momentum to line of sight) in radians.
        psi : array_like or float, default=0.0
            Polarization angle in radians.
        phase : array_like or float, default=0.0
            Coalescence phase in radians.
        geocent_time : array_like or float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        ra : array_like or float, default=0.0
            Right ascension in radians.
        dec : array_like or float, default=0.0
            Declination in radians.
        a_1 : array_like or float, default=0.0
            Primary spin magnitude (dimensionless).
        a_2 : array_like or float, default=0.0
            Secondary spin magnitude (dimensionless).
        tilt_1 : array_like or float, default=0.0
            Primary spin tilt angle in radians.
        tilt_2 : array_like or float, default=0.0
            Secondary spin tilt angle in radians.
        phi_12 : array_like or float, default=0.0
            Azimuthal angle between spins in radians.
        phi_jl : array_like or float, default=0.0
            Azimuthal angle between total and orbital angular momentum in radians.
        lambda_1 : array_like or float, default=0.0
            Primary tidal deformability (dimensionless).
        lambda_2 : array_like or float, default=0.0
            Secondary tidal deformability (dimensionless).
        eccentricity : array_like or float, default=0.0
            Orbital eccentricity at reference frequency.
        gw_param_dict : dict or bool, default=False
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are detector names 
            ('H1', 'L1', 'V1', etc.) and 'snr_net'. Values are arrays matching input size.

        Notes
        -----
        - For interpolation methods, tilt angles are converted to aligned spins: a_i * cos(tilt_i)
        - Total mass must be within [mtot_min, mtot_max] for non-zero SNR
        - Hybrid recalculation uses higher-order waveforms near detection threshold if enabled
        - Compatible with all configured detector networks and waveform approximants

        Examples
        --------
        >>> snr = GWSNR(snr_method='interpolation_no_spins')
        >>> result = snr.optimal_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
        >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")
        
        >>> # Multiple systems with parameter dictionary
        >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
        >>> result = snr.optimal_snr(gw_param_dict=params)
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
            "interpolation_no_spins_numba",
            "interpolation_aligned_spins",
            "interpolation_aligned_spins_numba",
            "interpolation_no_spins_jax",
            "interpolation_aligned_spins_jax",
            "interpolation_no_spins_mlx",
            "interpolation_aligned_spins_mlx",
        ]

        if self.snr_method in interpolation_list:

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

            snr_dict = self.optimal_snr_with_interpolation(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )
            gw_param_dict['a_1'] = a_1_old
            gw_param_dict['a_2'] = a_2_old

        elif self.snr_method == "inner_product":

            snr_dict = self.optimal_snr_with_inner_product(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )

        elif self.snr_method == "inner_product_jax":

            snr_dict = self.optimal_snr_with_inner_product_ripple(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )

        elif self.snr_method == "ann":
            snr_dict = self.optimal_snr_with_ann(
                gw_param_dict=gw_param_dict,
                output_jsonfile=output_jsonfile,
            )

        else:
            raise ValueError("SNR function type not recognised")
        

        if self.snr_recalculation:
            waveform_approximant_old = self.waveform_approximant
            self.waveform_approximant = self.snr_recalculation_waveform_approximant

            snr_net = snr_dict["snr_net"]
            min_, max_ = self.snr_recalculation_range
            idx = np.logical_and(
                snr_net >= min_,
                snr_net <= max_,
            )
            if np.sum(idx) != 0:
                print(
                    f"Recalculating SNR for {np.sum(idx)} out of {len(snr_net)} samples in the SNR range of {min_} to {max_}"
                )
                # print(f'\n length of idx: {len(idx)}')
                # print(f'\n length of tilt_2: {len(idx)}')
                input_dict = {}
                for key in gw_param_dict.keys():
                    input_dict[key] = np.array(gw_param_dict[key])[idx]

                snr_dict_ = self.optimal_snr_with_inner_product(
                    gw_param_dict=input_dict,
                )

                # iterate over detectors and update the snr_dict
                for key in snr_dict.keys():
                    if key in snr_dict_.keys():
                        snr_dict[key][idx] = snr_dict_[key]

            self.waveform_approximant = waveform_approximant_old
        
        return snr_dict

    def optimal_snr_with_ann(
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
        Calculate SNR using artificial neural network (ANN) prediction.

        Uses pre-trained neural networks to rapidly estimate optimal SNR for gravitational wave 
        signals with arbitrary spin configurations. The method first computes partial-scaled SNR 
        via interpolation, then feeds this along with other intrinsic parameters to detector-specific 
        ANN models for fast SNR prediction.

        Parameters
        ----------
        mass_1 : array_like or float, default=30.0
            Primary mass in solar masses.
        mass_2 : array_like or float, default=29.0
            Secondary mass in solar masses.
        luminosity_distance : array_like or float, default=100.0
            Luminosity distance in Mpc.
        theta_jn : array_like or float, default=0.0
            Inclination angle in radians.
        psi : array_like or float, default=0.0
            Polarization angle in radians.
        phase : array_like or float, default=0.0
            Coalescence phase in radians.
        geocent_time : array_like or float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        ra : array_like or float, default=0.0
            Right ascension in radians.
        dec : array_like or float, default=0.0
            Declination in radians.
        a_1 : array_like or float, default=0.0
            Primary spin magnitude (dimensionless).
        a_2 : array_like or float, default=0.0
            Secondary spin magnitude (dimensionless).
        tilt_1 : array_like or float, default=0.0
            Primary tilt angle in radians.
        tilt_2 : array_like or float, default=0.0
            Secondary tilt angle in radians.
        phi_12 : array_like or float, default=0.0
            Azimuthal angle between spins in radians.
        phi_jl : array_like or float, default=0.0
            Azimuthal angle between total and orbital angular momentum in radians.
        gw_param_dict : dict or bool, default=False
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR estimates for each detector and network. Keys are detector names 
            ('H1', 'L1', 'V1', etc.) and 'snr_net'.

        Notes
        -----
        - Requires pre-trained ANN models loaded during initialization
        - Uses aligned spin components: a_i * cos(tilt_i) for effective spin calculation
        - ANN inputs: partial-scaled SNR, amplitude factor, mass ratio, effective spin, inclination
        - Applies error correction to improve prediction accuracy
        - Total mass must be within [mtot_min, mtot_max] for valid results

        Examples
        --------
        >>> snr = GWSNR(snr_method='ann')
        >>> result = snr.optimal_snr_with_ann(mass_1=30, mass_2=25, a_1=0.5, tilt_1=0.2)
        >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")
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
        optimal_snr["snr_net"] = np.zeros(size)
        for i, det in enumerate(detectors):
            x = scaler[det].transform(ann_input[i])
            optimal_snr_ = model[det].predict(x, verbose=0).flatten()
            # adjusting the optimal SNR with error adjustment
            optimal_snr[det][idx_tracker] = optimal_snr_ - (self.error_adjustment[det]['slope']*optimal_snr_ + self.error_adjustment[det]['intercept'])
            optimal_snr["snr_net"] += optimal_snr[det] ** 2
        optimal_snr["snr_net"] = np.sqrt(optimal_snr["snr_net"])

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json(output_filename, optimal_snr)

        return optimal_snr

    def output_ann(self, idx, params):
        """
        Prepare ANN input features from gravitational wave parameters.

        Transforms gravitational wave parameters into feature vectors for neural network 
        prediction. Calculates partial-scaled SNR via interpolation and combines with 
        intrinsic parameters to create standardized input features.

        Parameters
        ----------
        idx : numpy.ndarray of bool
            Boolean mask for valid mass ranges (mtot_min <= mtot <= mtot_max).
        params : dict
            GW parameter dictionary with keys: mass_1, mass_2, luminosity_distance,
            theta_jn, a_1, a_2, tilt_1, tilt_2, psi, geocent_time, ra, dec.

        Returns
        -------
        list of numpy.ndarray
            Feature arrays for each detector, shape (N, 5) with columns:
            [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn].

        Notes
        -----
        - Uses aligned spin components: a_i * cos(tilt_i)
        - Amplitude factor: A1 = Mc^(5/6) / d_eff
        - Effective spin: chi_eff = (m1*a1z + m2*a2z) / (m1+m2)
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

    def optimal_snr_with_interpolation(
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
        Calculate SNR (for non-spinning or aligned-spin) using bicubic interpolation of precomputed coefficients.

        Fast SNR calculation method using interpolated partial-scaled SNR values across
        intrinsic parameter grids. Supports no-spin and aligned-spin configurations with
        Numba or JAX acceleration for population studies.

        Parameters
        ----------
        mass_1 : array_like or float, default=30.0
            Primary mass in solar masses.
        mass_2 : array_like or float, default=29.0
            Secondary mass in solar masses.
        luminosity_distance : array_like or float, default=100.0
            Luminosity distance in Mpc.
        theta_jn : array_like or float, default=0.0
            Inclination angle in radians.
        psi : array_like or float, default=0.0
            Polarization angle in radians.
        phase : array_like or float, default=0.0
            Coalescence phase in radians.
        geocent_time : array_like or float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        ra : array_like or float, default=0.0
            Right ascension in radians.
        dec : array_like or float, default=0.0
            Declination in radians.
        a_1 : array_like or float, default=0.0
            Primary aligned spin component (for aligned-spin methods only).
        a_2 : array_like or float, default=0.0
            Secondary aligned spin component (for aligned-spin methods only).
        gw_param_dict : dict or bool, default=False
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are detector names
            ('H1', 'L1', 'V1', etc.) and 'snr_net'. Systems outside mass bounds have zero SNR.

        Notes
        -----
        - Requires precomputed interpolation coefficients from class initialization
        - self.get_interpolated_snr is set based on snr_method (Numba or JAX or MLX) and whether the system is non-spinning or aligned-spin
        - Total mass must be within [mtot_min, mtot_max] for valid results
        - Uses aligned spin: a_i * cos(tilt_i) for spin-enabled methods
        - Backend acceleration available via JAX or Numba depending on snr_method

        Examples
        --------
        >>> snr_calc = GWSNR(snr_method='interpolation_no_spins')
        >>> result = snr_calc.optimal_snr_with_interpolation(mass_1=30, mass_2=25)
        >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")
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
        if self.mtot_cut:
            idx2 = np.logical_and(mtot >= self.mtot_min, mtot <= self.mtot_max)
        else:
            idx2 = np.ones_like(mtot, dtype=bool)
        idx_tracker = np.nonzero(idx2)[0]

        # Set multiprocessing start method to 'spawn' for multri-threading compatibility
        # mp.set_start_method('spawn', force=True)

        # Get interpolated SNR
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
            int(self.batch_size_interpolation),
        )

        # Create optimal_snr dictionary using dictionary comprehension
        optimal_snr = {det: np.zeros(size) for det in detectors}
        optimal_snr["snr_net"] = np.zeros(size)
        for j, det in enumerate(detectors):
            optimal_snr[det][idx_tracker] = snr[j]
        optimal_snr["snr_net"][idx_tracker] = snr_effective

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json(output_filename, optimal_snr)

        return optimal_snr
    
    def init_partialscaled(self):
        """
        Generate partial-scaled SNR interpolation coefficients for fast bicubic interpolation.

        Computes and saves distance-independent SNR coefficients across intrinsic parameter grids
        for each detector. These coefficients enable fast runtime SNR calculation via interpolation
        without requiring waveform generation.

        Creates parameter grids based on interpolation method:
        - No-spin: 2D grid (mass_ratio, total_mass) 
        - Aligned-spin: 4D grid (mass_ratio, total_mass, a_1, a_2)

        For each grid point, computes optimal SNR with fixed extrinsic parameters 
        (d_L=100 Mpc, θ_jn=0, overhead sky location), then scales by effective distance 
        and chirp mass: partial_SNR = (optimal_SNR × d_eff) / Mc^(5/6).

        Coefficients are saved as pickle files for runtime interpolation.

        Raises
        ------
        ValueError
            If mtot_min < 1.0 or snr_method not supported for interpolation.

        Notes
        -----
        Grid dimensions set by ratio_resolution, mtot_resolution, spin_resolution.
        Automatically called during initialization when coefficients missing.
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

        
        list_no_spins = ["interpolation", "interpolation_no_spins", "interpolation_no_spins_numba", "interpolation_no_spins_jax", "interpolation_no_spins_mlx"]
        list_aligned_spins = ["interpolation_aligned_spins", "interpolation_aligned_spins_numba", "interpolation_aligned_spins_jax", "interpolation_aligned_spins_mlx"]

        # Create broadcastable 4D grids
        if self.snr_method in list_aligned_spins:
            a_1_table = self.a_1_arr.copy()
            a_2_table = self.a_2_arr.copy()
            size3 = self.spin_resolution
            size4 = self.spin_resolution
            a_1_table = np.asarray(a_1_table)
            a_2_table = np.asarray(a_2_table)

            q, mtot, a_1, a_2 = np.meshgrid(
                ratio_table, mtot_table, a_1_table, a_2_table, indexing='ij'
            )
        elif self.snr_method  in list_no_spins:
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
        optimal_snr_unscaled = self.optimal_snr_with_inner_product(
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

        for j in num_det:
            if self.snr_method in list_aligned_spins:
                snr_partial_ = np.array(np.reshape(optimal_snr_unscaled[detectors[j]],(size1, size2, size3, size4)) * dl_eff[j] / Mchirp_scaled, dtype=np.float32), # shape (size1, size2, size3, size4)
            elif self.snr_method in list_no_spins:
                snr_partial_ = np.array(np.reshape(optimal_snr_unscaled[detectors[j]],(size1, size2)) * dl_eff[j] / Mchirp_scaled, dtype=np.float32), # shape (size1, size2, size3, size4)
            else:
                raise ValueError(f"snr_method {self.snr_method} is not supported for interpolation.")
            # print('dl_eff=',dl_eff[j])
            # print('Mchirp_scaled=',Mchirp_scaled.shape)
            # print('optimal_snr_unscaled=',np.reshape(optimal_snr_unscaled[detectors[j]],(size1, size2, size3, size4)).shape)
            print(f"\nSaving Partial-SNR for {detectors[j]} detector with shape {snr_partial_[0].shape}")
            save_pickle(self.path_interpolator[j], snr_partial_[0])

    def optimal_snr_with_inner_product(
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
        Calculate optimal SNR using LAL waveform generation and noise-weighted inner products.

        This method computes SNR by generating gravitational wave signals with LAL and calculating
        matched filtering inner products against detector noise PSDs. Supports all LAL waveform
        approximants including aligned and precessing spin systems.

        Parameters
        ----------
        mass_1 : array_like or float, default=10
            Primary mass in solar masses.
        mass_2 : array_like or float, default=10
            Secondary mass in solar masses.
        luminosity_distance : array_like or float, default=100.0
            Luminosity distance in Mpc.
        theta_jn : array_like or float, default=0.0
            Inclination angle in radians.
        psi : array_like or float, default=0.0
            Polarization angle in radians.
        phase : array_like or float, default=0.0
            Coalescence phase in radians.
        geocent_time : array_like or float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        ra : array_like or float, default=0.0
            Right ascension in radians.
        dec : array_like or float, default=0.0
            Declination in radians.
        a_1 : array_like or float, default=0.0
            Primary spin magnitude (dimensionless).
        a_2 : array_like or float, default=0.0
            Secondary spin magnitude (dimensionless).
        tilt_1 : array_like or float, default=0.0
            Primary spin tilt angle in radians.
        tilt_2 : array_like or float, default=0.0
            Secondary spin tilt angle in radians.
        phi_12 : array_like or float, default=0.0
            Azimuthal angle between spins in radians.
        phi_jl : array_like or float, default=0.0
            Azimuthal angle between total and orbital angular momentum in radians.
        lambda_1 : array_like or float, default=0.0
            Primary tidal deformability (dimensionless).
        lambda_2 : array_like or float, default=0.0
            Secondary tidal deformability (dimensionless).
        eccentricity : array_like or float, default=0.0
            Orbital eccentricity at reference frequency.
        gw_param_dict : dict or bool, default=False
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are detector names 
            ('H1', 'L1', 'V1', etc.) and 'snr_net'. Systems outside mass bounds have zero SNR.

        Notes
        -----
        - Waveform duration auto-estimated from chirp time with 1.1x safety factor
        - Uses multiprocessing for parallel computation across npool processors
        - Requires 'if __name__ == "__main__":' guard when using multiprocessing
        - Most accurate method but slower than interpolation for population studies

        Examples
        --------
        >>> snr = GWSNR(snr_method='inner_product')
        >>> result = snr.optimal_snr_with_inner_product(mass_1=30, mass_2=25)
        >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")
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
        f_ref = self.f_ref
        if f_ref is None:
            f_ref = f_min
        num_det = np.arange(len(detectors), dtype=int)

        # get the psds for the required detectors
        # psd_dict = {detectors[i]: self.psds_list[i] for i in num_det}
        psd_list = self.psds_list.copy()
        num = len(mass_1)

        #############################################
        # setting up parameters for multiprocessing #
        #############################################
        mtot = mass_1 + mass_2
        if self.mtot_cut:
            idx = np.logical_and(mtot >= self.mtot_min, mtot <= self.mtot_max)
        else:
            idx = np.ones_like(mtot, dtype=bool)
        size1 = np.sum(idx)
        iterations = np.arange(size1)  # to keep track of index

        # IMPORTANT: time duration calculation for each of the mass combination
        if not self.fixed_duration:
            safety = 1.1
            approx_duration = safety * findchirp_chirptime(mass_1[idx], mass_2[idx], f_min)
            duration = np.ceil(approx_duration + 2.0)
            if self.duration_max:
                duration[duration > self.duration_max] = self.duration_max  # IMRPheonomXPHM has maximum duration of 371s
            if self.duration_min:
                duration[duration < self.duration_min] = self.duration_min
        else:
            duration = self.fixed_duration * np.ones_like(mass_1[idx])

        frequency_domain_source_model = self.frequency_domain_source_model

        # get polarization tensor
        # np.shape(Fp) = (size1, len(num_det))
        Fp, Fc = antenna_response_array(
            ra[idx], dec[idx], geocent_time[idx], psi[idx], detector_tensor
        )

        # Set up input arguments for multiprocessing
        input_arguments = [(
            mass_1_i,
            mass_2_i,
            luminosity_distance_i,
            theta_jn_i,
            psi_i,
            phase_i,
            ra_i,
            dec_i,
            geocent_time_i,
            a_1_i,
            a_2_i,
            tilt_1_i,
            tilt_2_i,
            phi_12_i,
            phi_jl_i,
            lambda_1_i,
            lambda_2_i,
            eccentricity_i,
            approximant,
            f_min,
            f_ref,
            duration_i,
            sampling_frequency,
            iterations_i,
            psd_list,
            frequency_domain_source_model,
            ) for (mass_1_i, mass_2_i, luminosity_distance_i, theta_jn_i, psi_i, phase_i, ra_i, dec_i, geocent_time_i, a_1_i, a_2_i, tilt_1_i, tilt_2_i, phi_12_i, phi_jl_i, lambda_1_i, lambda_2_i, eccentricity_i, duration_i, iterations_i) in zip(
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
                duration,
                iterations,
            )
        ]

        # np.shape(hp_inner_hp) = (len(num_det), size1)
        hp_inner_hp = np.zeros((len(num_det), size1), dtype=np.complex128)
        hc_inner_hc = np.zeros((len(num_det), size1), dtype=np.complex128)

        if self.snr_type=='optimal_snr':

            self._multiprocessing_error()
            with mp.Pool(processes=npool) as pool:
                # call the same function with different data in parallel
                # imap->retain order in the list, while map->doesn't
                if self.multiprocessing_verbose:
                    for result in tqdm(
                        pool.imap_unordered(noise_weighted_inner_prod_h_inner_h, input_arguments),
                        total=len(input_arguments),
                        ncols=100,
                    ):
                        # but, np.shape(hp_inner_hp_i) = (size1, len(num_det))
                        hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                        hp_inner_hp[:, iter_i] = hp_inner_hp_i
                        hc_inner_hc[:, iter_i] = hc_inner_hc_i
                else:
                    # with map, without tqdm
                    for result in pool.map(noise_weighted_inner_prod_h_inner_h, input_arguments):
                        hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                        hp_inner_hp[:, iter_i] = hp_inner_hp_i
                        hc_inner_hc[:, iter_i] = hc_inner_hc_i

            # combining the results
            snrs_sq = abs((Fp**2) * hp_inner_hp + (Fc**2) * hc_inner_hc)
            snr = np.sqrt(snrs_sq)

        elif self.snr_type=='observed_snr':

            raise ValueError("observed_snr not implemented yet")
        
        else:
            raise ValueError("snr_type should be either 'optimal_snr' or 'observed_snr'")
        
        snr_effective = np.sqrt(np.sum(snr**2, axis=0))

        # organizing the snr dictionary
        optimal_snr = dict()
        for j, det in enumerate(detectors):
            snr_buffer = np.zeros(num, dtype=np.complex128)
            snr_buffer[idx] = snr[j]
            optimal_snr[det] = snr_buffer
        snr_buffer = np.zeros(num, dtype=np.complex128)
        snr_buffer[idx] = snr_effective
        optimal_snr["snr_net"] = snr_buffer

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json(output_filename, optimal_snr)

        return optimal_snr
    
    def _multiprocessing_error(self):
        """
        Prints an error message when multiprocessing is used.
        """
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

    def optimal_snr_with_inner_product_ripple(
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
        Calculate optimal SNR using JAX-accelerated Ripple waveforms and noise-weighted inner products.

        Uses the Ripple waveform generator with JAX backend for fast SNR computation via 
        vectorized inner products. Supports arbitrary spin configurations and provides 
        significant speedup over LAL-based methods for population studies.

        Parameters
        ----------
        mass_1 : array_like or float, default=10
            Primary mass in solar masses.
        mass_2 : array_like or float, default=10
            Secondary mass in solar masses.
        luminosity_distance : array_like or float, default=100.0
            Luminosity distance in Mpc.
        theta_jn : array_like or float, default=0.0
            Inclination angle in radians.
        psi : array_like or float, default=0.0
            Polarization angle in radians.
        phase : array_like or float, default=0.0
            Coalescence phase in radians.
        geocent_time : array_like or float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        ra : array_like or float, default=0.0
            Right ascension in radians.
        dec : array_like or float, default=0.0
            Declination in radians.
        a_1 : array_like or float, default=0.0
            Primary spin magnitude (dimensionless).
        a_2 : array_like or float, default=0.0
            Secondary spin magnitude (dimensionless).
        tilt_1 : array_like or float, default=0.0
            Primary spin tilt angle in radians.
        tilt_2 : array_like or float, default=0.0
            Secondary spin tilt angle in radians.
        phi_12 : array_like or float, default=0.0
            Azimuthal angle between spins in radians.
        phi_jl : array_like or float, default=0.0
            Azimuthal angle between total and orbital angular momentum in radians.
        lambda_1 : array_like or float, default=0.0
            Primary tidal deformability (dimensionless).
        lambda_2 : array_like or float, default=0.0
            Secondary tidal deformability (dimensionless).
        eccentricity : array_like or float, default=0.0
            Orbital eccentricity at reference frequency.
        gw_param_dict : dict or bool, default=False
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are detector names 
            ('H1', 'L1', 'V1', etc.) and 'snr_net'. Systems outside mass bounds have zero SNR.

        Notes
        -----
        - Requires snr_method='inner_product_jax' during initialization
        - Uses JAX JIT compilation and vectorization for GPU acceleration
        - Duration auto-estimated with safety bounds from duration_min/max
        - Compatible with Ripple-supported approximants (IMRPhenomD, IMRPhenomXPHM)
        - Supports precessing spins through full parameter space

        Examples
        --------
        >>> snr = GWSNR(snr_method='inner_product_jax')
        >>> result = snr.optimal_snr_with_inner_product_ripple(mass_1=30, mass_2=25)
        >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")
        """

        # if gw_param_dict is given, then use that
        if gw_param_dict is not False:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, eccentricity  = get_gw_parameters(gw_param_dict)
        else:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, eccentricity = get_gw_parameters(dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, ra=ra, dec=dec, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl, lambda_1=lambda_1, lambda_2=lambda_2, eccentricity=eccentricity))

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
        if self.mtot_cut:
            idx = np.logical_and(mtot >= self.mtot_min, mtot <= self.mtot_max)
        else:
            idx = np.ones_like(mtot, dtype=bool)
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
            lambda_1=lambda_1[idx],
            lambda_2=lambda_2[idx],
            eccentricity=eccentricity[idx],
        )

        # from ripple_class.noise_weighted_inner_product_jax
        hp_inner_hp, hc_inner_hc = self.noise_weighted_inner_product_jax(
            gw_param_dict=input_dict, 
            psd_list=psd_list,
            detector_list=detectors, 
            duration_min=self.duration_min,
            duration_max=self.duration_max,
            npool=npool,
            multiprocessing_verbose=self.multiprocessing_verbose
        )

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
        optimal_snr["snr_net"] = snr_buffer

        # Save as JSON file
        if output_jsonfile:
            output_filename = (output_jsonfile if isinstance(output_jsonfile, str) else "snr.json")
            save_json(output_filename, optimal_snr)

        return optimal_snr

    def pdet(
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
        snr_th=None,
        snr_th_net=None,
        pdet_type=None,
        distribution_type=None,
    ):
        """
        Calculate probability of detection for gravitational wave signals.

        Computes detection probability based on SNR thresholds for individual detectors and detector networks. Accounts for noise fluctuations by modeling observed SNR as statistical distributions around optimal SNR values.

        Parameters
        ----------
        mass_1 : array_like or float, default=np.array([10.0])
            Primary mass in solar masses.
        mass_2 : array_like or float, default=np.array([10.0])
            Secondary mass in solar masses.
        luminosity_distance : array_like or float, default=100.0
            Luminosity distance in Mpc.
        theta_jn : array_like or float, default=0.0
            Inclination angle in radians.
        psi : array_like or float, default=0.0
            Polarization angle in radians.
        phase : array_like or float, default=0.0
            Coalescence phase in radians.
        geocent_time : array_like or float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        ra : array_like or float, default=0.0
            Right ascension in radians.
        dec : array_like or float, default=0.0
            Declination in radians.
        a_1 : array_like or float, default=0.0
            Primary spin magnitude (dimensionless).
        a_2 : array_like or float, default=0.0
            Secondary spin magnitude (dimensionless).
        tilt_1 : array_like or float, default=0.0
            Primary spin tilt angle in radians.
        tilt_2 : array_like or float, default=0.0
            Secondary spin tilt angle in radians.
        phi_12 : array_like or float, default=0.0
            Azimuthal angle between spins in radians.
        phi_jl : array_like or float, default=0.0
            Azimuthal angle between total and orbital angular momentum in radians.
        lambda_1 : array_like or float, default=0.0
            Primary tidal deformability (dimensionless).
        lambda_2 : array_like or float, default=0.0
            Secondary tidal deformability (dimensionless).
        eccentricity : array_like or float, default=0.0
            Orbital eccentricity at reference frequency.
        gw_param_dict : dict or bool, default=False
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'pdet.json'.
        snr_th : float, array_like, or None, default=None
            SNR threshold for individual detectors. If None, uses pdet_kwargs['snr_th'].
            If array, must match number of detectors.
        snr_th_net : float or None, default=None
            Network SNR threshold. If None, uses pdet_kwargs['snr_th_net'].
        pdet_type : str or None, default=None
            Detection probability method:
            - 'boolean': Binary detection (0 or 1) based on noise realizations
            - 'probability_distribution': Analytical probability using noise statistics
            If None, uses pdet_kwargs['pdet_type'].
        distribution_type : str or None, default=None
            Noise model for observed SNR:
            - 'gaussian': Gaussian noise (sigma=1)
            - 'noncentral_chi2': Non-central chi-squared (2 DOF per detector)
            If None, uses pdet_kwargs['distribution_type'].

        Returns
        -------
        dict
            Detection probabilities for each detector and network. Keys are detector 
            names ('H1', 'L1', 'V1', etc.) and 'pdet_net'. Values depend on pdet_type:
            - 'boolean': Binary arrays (0/1) indicating detection
            - 'probability_distribution': Probability arrays (0-1)

        Notes
        -----
        - First computes optimal SNR using configured snr_method
        - Models observed SNR as noisy version of optimal SNR
        - Non-central chi-squared uses 2 DOF per detector, network uses 2×N_det DOF
        - Boolean method generates random noise realizations for each system
        - Probability method uses analytical CDFs for faster computation

        Examples
        --------
        >>> pdet_calc = GWSNR(pdet_kwargs={'snr_th': 8, 'pdet_type': 'boolean'})
        >>> result = pdet_calc.pdet(mass_1=30, mass_2=25, luminosity_distance=200)
        >>> print(f"Network detection: {result['pdet_net'][0]}")
        
        >>> # Analytical probability calculation
        >>> pdet_calc = GWSNR(pdet_kwargs={'pdet_type': 'probability_distribution'})
        >>> probs = pdet_calc.pdet(mass_1=[20,30], mass_2=[20,25], luminosity_distance=150)
        """

        # get SNR first 
        snr_dict = self.optimal_snr(
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
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            eccentricity=eccentricity,
            gw_param_dict=gw_param_dict,
            output_jsonfile=output_jsonfile,
        )

        snr_th = snr_th if snr_th else self.pdet_kwargs["snr_th"]
        snr_th_net = snr_th_net if snr_th_net else self.pdet_kwargs["snr_th_net"]
        pdet_type = pdet_type if pdet_type else self.pdet_kwargs["pdet_type"]
        distribution_type = distribution_type if distribution_type else self.pdet_kwargs["distribution_type"]

        # check if snr_th is an array (for multi-detector) or a single value
        if isinstance(snr_th, (list, np.ndarray)):
            snr_th = np.array(snr_th)
        else:
            snr_th = np.full(len(self.detector_list), snr_th)

        detectors = np.array(self.detector_list)
        pdet_dict = {}
        for i, det in enumerate(detectors):
            if pdet_type == "probability_distribution":
                if distribution_type == "noncentral_chi2":
                    df = 2  # 2 quadratures per IFO
                    nc_param = snr_dict[det]**2  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                    # sum up the probabilities from snr_th to inf
                    pdet_dict[det] = 1 - ncx2.cdf(snr_th[i]**2, df=df, nc=nc_param)

                elif distribution_type == "gaussian":
                    pdet_dict[det] = 1 - norm.cdf(snr_th[i] - snr_dict[det])

            elif pdet_type == "boolean":
                if distribution_type == "noncentral_chi2":

                    df = 2  # 2 quadratures per IFO
                    nc_param = snr_dict[det]**2  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                    # sum up the probabilities from snr_th to inf
                    observed_snr = np.sqrt(ncx2.rvs(df=df, nc=nc_param, size=snr_dict[det].shape))

                    pdet_dict[det] = np.array(snr_th[i] < observed_snr, dtype=int)

                elif distribution_type == "gaussian":

                    observed_snr = snr_dict[det] + np.random.normal(0, 1, size=snr_dict[det].shape)
                    pdet_dict[det] = np.array(snr_th[i] < observed_snr, dtype=int)

            else:
                raise ValueError("pdet_type should be either 'boolean' or 'probability_distribution'")
            
        # for network
        if pdet_type == "probability_distribution":

            if distribution_type == "noncentral_chi2":
                df = 2 * len(detectors)  # 2 quadratures per IFO
                nc_param = snr_dict["snr_net"]**2  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                # sum up the probabilities from snr_th to inf
                pdet_dict["pdet_net"] = 1 - ncx2.cdf(snr_th_net**2, df=df, nc=nc_param)
            elif distribution_type == "gaussian":
                pdet_dict["pdet_net"] = np.array(1 - norm.cdf(snr_th_net - snr_dict["snr_net"]))

        elif pdet_type == "boolean":
            if distribution_type == "noncentral_chi2":
                df = 2 * len(detectors)  # 2 quadratures per IFO
                nc_param = snr_dict["snr_net"]**2  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                # sum up the probabilities from snr_th to inf
                observed_snr_net = np.sqrt(ncx2.rvs(df=df, nc=nc_param, size=snr_dict["snr_net"].shape))
                pdet_dict["pdet_net"] = np.array(snr_th_net < observed_snr_net, dtype=int)
            elif distribution_type == "gaussian":
                observed_snr_net = snr_dict["snr_net"] + np.random.normal(0, 1, size=snr_dict["snr_net"].shape)
                pdet_dict["pdet_net"] = np.array(snr_th_net < observed_snr_net, dtype=int)

        return pdet_dict

    def horizon_distance_analytical(self, mass_1=1.4, mass_2=1.4, snr_th=None, snr_th_net=None):
        """
        Calculate detector horizon distance for compact binary coalescences.
        
        Computes the maximum range at which a source can be detected with optimal 
        orientation (face-on, overhead). Uses reference SNR at 100 Mpc scaled by 
        effective distance and detection threshold.

        Parameters
        ----------
        mass_1 : array_like or float, default=1.4
            Primary mass in solar masses.
        mass_2 : array_like or float, default=1.4
            Secondary mass in solar masses.
        snr_th : float, optional
            Individual detector SNR threshold. Uses class default if None.
        snr_th_net : float, optional
            Network SNR threshold. Uses class default if None.

        Returns
        -------
        dict
            Horizon distances in Mpc for each detector and network.
            Keys: detector names ('H1', 'L1', etc.) and 'snr_net'.

        Notes
        -----
        - Assumes optimal orientation: θ_jn=0, overhead sky location
        - Formula: d_horizon = (d_eff/SNR_th) × SNR_100Mpc
        - Network horizon uses quadrature sum of detector responses
        - Compatible with all waveform approximants

        Examples
        --------
        >>> snr = GWSNR(snr_method='inner_product')
        >>> horizon = snr.horizon_distance_analytical(mass_1=1.4, mass_2=1.4)
        >>> print(f"H1 horizon: {horizon['H1']:.1f} Mpc")
        """

        from ..numba import effective_distance

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
        luminosity_distance_ = 1000.0

        # calling bilby_snr
        optimal_snr_unscaled = self.optimal_snr(
            mass_1=mass_1,
            mass_2=mass_2,
            luminosity_distance=luminosity_distance_,
            theta_jn=theta_jn_,
            psi=psi_,
            phase=phase_,
            ra=ra_,
            dec=dec_,
        )

        horizon = dict.fromkeys(detectors, 0.0)
        dl_eff = np.zeros(len(detectors), dtype=float)
        for j, det in enumerate(detectors):
            # get the effective distance for each detector
            dl_eff[j] = effective_distance(
                luminosity_distance=luminosity_distance_,
                theta_jn=theta_jn_,
                ra=ra_, 
                dec=dec_,
                geocent_time=geocent_time_,
                psi=psi_,
                detector_tensor=detector_tensor[j]
            )

            # Horizon calculation
            horizon[det] = (dl_eff[j] / snr_th) * optimal_snr_unscaled[det]

        return horizon

    def horizon_distance_numerical(self, 
            mass_1=1.4, 
            mass_2=1.4,
            psi=0.0,
            phase=0.0,
            geocent_time=1246527224.169434,
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
            snr_th=None, 
            snr_th_net=None,
            detector_location_as_optimal_sky=False,
            minimize_function_dict=None, 
            root_scalar_dict=None,
            maximization_check=False,
        ):
        """
        Calculate detector horizon distance with optimal sky positioning and arbitrary spin parameters.

        Finds the maximum luminosity distance at which a gravitational wave signal can be 
        detected above threshold SNR. For each detector, determines optimal sky location
        that maximizes antenna response, then solves for distance where SNR equals threshold.

        Parameters
        ----------
        mass_1 : array_like or float, default=1.4
            Primary mass in solar masses.
        mass_2 : array_like or float, default=1.4
            Secondary mass in solar masses.
        psi : array_like or float, default=0.0
            Polarization angle in radians.
        phase : array_like or float, default=0.0
            Coalescence phase in radians.
        geocent_time : float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        a_1 : array_like or float, default=0.0
            Primary spin magnitude (dimensionless).
        a_2 : array_like or float, default=0.0
            Secondary spin magnitude (dimensionless).
        tilt_1 : array_like or float, default=0.0
            Primary spin tilt angle in radians.
        tilt_2 : array_like or float, default=0.0
            Secondary spin tilt angle in radians.
        phi_12 : array_like or float, default=0.0
            Azimuthal angle between spins in radians.
        phi_jl : array_like or float, default=0.0
            Azimuthal angle between total and orbital angular momentum in radians.
        lambda_1 : array_like or float, default=0.0
            Primary tidal deformability (dimensionless).
        lambda_2 : array_like or float, default=0.0
            Secondary tidal deformability (dimensionless).
        eccentricity : array_like or float, default=0.0
            Orbital eccentricity at reference frequency.
        gw_param_dict : dict or bool, default=False
            Parameter dictionary. If provided, overrides individual arguments.
        snr_th : float, optional
            Individual detector SNR threshold. Uses class default if None.
        snr_th_net : float, optional
            Network SNR threshold. Uses class default if None.
        detector_location_as_optimal_sky : bool, default=False
            If True, uses detector zenith as optimal sky location instead of optimization.
        minimize_function_dict : dict, optional
            Parameters for sky location optimization. It contains input for scipy's differential_evolution. 
            Default: dict(
                bounds=[(0, 2*np.pi), (-np.pi/2, np.pi/2)], # ra, dec bounds
                tol=1e-7, 
                polish=True, 
                maxiter=10000
            )
        root_scalar_dict : dict, optional
            Parameters for horizon distance root finding. It contains input for scipy's root_scalar. 
            Default: dict(
                bracket=[1, 100000], # redshift range
                method='bisect',
                xtol=1e-5
            )
        maximization_check : bool, default=False
            Verify that antenna response maximization achieved ~1.0.

        Returns
        -------
        horizon : dict
            Horizon distances in Mpc for each detector and network ('snr_net').
        sky_location : dict
            Optimal sky coordinates (ra, dec) in radians for maximum SNR at given geocent_time.

        Notes
        -----
        - Uses differential evolution to find optimal sky location maximizing antenna response
        - Network horizon maximizes quadrature sum of detector SNRs
        - Individual detector horizons maximize (F_plus² + F_cross²) 
        - Root finding determines distance where SNR equals threshold
        - Computation time depends on optimization tolerances and system complexity

        Examples
        --------
        >>> snr = GWSNR(snr_method='inner_product')
        >>> horizon, sky = snr.horizon_distance_numerical(mass_1=1.4, mass_2=1.4)
        >>> print(f"Network horizon: {horizon['snr_net']:.1f} Mpc at (RA={sky['snr_net'][0]:.2f}, Dec={sky['snr_net'][1]:.2f})")
        """

        # find root i.e. snr = snr_th
        from scipy.optimize import root_scalar
        from scipy.optimize import differential_evolution
        from astropy.time import Time
        from astropy.coordinates import SkyCoord, AltAz, EarthLocation
        import astropy.units as u

        if gw_param_dict is not False:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, eccentricity = get_gw_parameters(gw_param_dict)
        else:
            mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, eccentricity = get_gw_parameters(dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, ra=ra, dec=dec, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl, lambda_1=lambda_1, lambda_2=lambda_2, eccentricity=eccentricity))

        snr_th = snr_th if snr_th else self.pdet_kwargs["snr_th"]
        snr_th_net = snr_th_net if snr_th_net else self.pdet_kwargs["snr_th_net"]

        if minimize_function_dict is None:
            minimize_function_dict = dict(
                bounds=[(0, 2*np.pi), (-np.pi/2, np.pi/2)],
                tol=1e-7, 
                polish=True, 
                maxiter=10000
            )

        if root_scalar_dict is None:
            root_scalar_dict = dict(
                bracket=[1, 100000], 
                method='bisect',
                xtol=1e-5
            )

        detectors = self.detector_list.copy()
        detectors.append("snr_net")
        detectors = np.array(detectors)

        horizon = dict.fromkeys(detectors, 0.0)
        sky_location = dict.fromkeys(detectors, (0.0, 0.0))  # ra, dec

        if detector_location_as_optimal_sky:
            ifos = self.ifos

        for i, det in enumerate(detectors):

            # minimize this function
            # do for network detectors first
            if det == "snr_net":
                def snr_minimize(x):
                    """ Minimize the inverse of SNR to maximize SNR """
                    ra, dec = x

                    snr = self.optimal_snr(
                        mass_1=mass_1, 
                        mass_2=mass_2, 
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
                    )['snr_net'][0]

                    return 1/snr

                # Use differential evolution to find the ra and dec that maximize the antenna response
                ra_max, dec_max = differential_evolution(
                    snr_minimize,
                    bounds= minimize_function_dict['bounds'],  # Only ra and dec bounds
                    tol=minimize_function_dict['tol'],
                    polish=minimize_function_dict['polish'],
                    maxiter=minimize_function_dict['maxiter']
                ).x
                
            else:
                # for individual detectors, find the sky location that maximizes (F_plus^2 + F_cross^2)
                if detector_location_as_optimal_sky:
                    # use astropy to find the zenith location of the detector at the given geocentric time
                    t = Time(geocent_time, format='gps', scale='utc')
                    loc = EarthLocation(lat=ifos[i].latitude*u.deg, lon=ifos[i].longitude*u.deg, height=ifos[i].elevation*u.m)
                    zen = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=AltAz(location=loc, obstime=t)).icrs
                    ra_max = zen.ra.rad
                    dec_max = zen.dec.rad

                else:
                    # use the maximization function to find the ra and dec that maximize the antenna response
                    def antenna_response_minimization(x):
                        ra, dec = x
                        f_plus = antenna_response_plus(ra, dec, geocent_time, psi, self.detector_tensor_list[i])
                        f_cross = antenna_response_cross(ra, dec, geocent_time, psi, self.detector_tensor_list[i])
                        return 1/(f_plus**2 + f_cross**2)

                    ra_max, dec_max = differential_evolution(
                        antenna_response_minimization,
                        bounds= minimize_function_dict['bounds'],  # Only ra and dec bounds
                        tol=minimize_function_dict['tol'],
                        polish=minimize_function_dict['polish'],
                        maxiter=minimize_function_dict['maxiter']
                    ).x

                # check the maximum antenna response
                if maximization_check is True:
                    
                    f_plus_max = antenna_response_plus(ra_max, dec_max, geocent_time, psi, self.detector_tensor_list[i])
                    f_cross_max = antenna_response_cross(ra_max, dec_max, geocent_time, psi, self.detector_tensor_list[i])
                    antenna_max = np.sqrt(f_plus_max**2 + f_cross_max**2)
                    # raise warning if antenna response is not close to 1
                    if not np.isclose(antenna_max, 1.0, atol=1e-2):
                        print(f"\n[WARNING] Maximum antenna response for {det} is {antenna_max:.3f}, which is not close to 1.0. The horizon distance may be underestimated.\n"
                            "This could be due to the chosen geocentric time or detector configuration.\n"
                            "Consider changing the geocentric time or checking the detector tensor.\n")

            # det = "snr_net"
            self.multiprocessing_verbose = False
            def snr_fn(dl):
                # optimal_snr_with_inner_product returns a dictionary with keys as detectors
                # and values as SNR values
                return self.optimal_snr(
                        mass_1=mass_1,
                        mass_2=mass_2,
                        luminosity_distance=dl,
                        psi=psi,
                        phase=phase,
                        geocent_time=geocent_time,
                        ra=ra_max,
                        dec=dec_max,
                        a_1=a_1,
                        a_2=a_2,
                        tilt_1=tilt_1,
                        tilt_2=tilt_2,
                        phi_12=phi_12,
                        phi_jl=phi_jl,
                        lambda_1=lambda_1,
                        lambda_2=lambda_2,
                        eccentricity=eccentricity,
                    )[det][0] - snr_th_net
            
            # find root i.e. snr = snr_th
            horizon[det] = root_scalar(
                snr_fn, 
                bracket=root_scalar_dict['bracket'], 
                method=root_scalar_dict['method'], 
                xtol=root_scalar_dict['xtol'],
            ).root
            sky_location[det] = (ra_max, dec_max)

        return horizon, sky_location
    

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