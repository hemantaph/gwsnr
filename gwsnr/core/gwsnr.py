# -*- coding: utf-8 -*-
"""
This module provides efficient calculation of signal-to-noise ratio (SNR) and probability of detection (Pdet) for gravitational wave signals from compact binary coalescences.

The module supports multiple computational backends including interpolation-based methods for fast calculation,
inner product methods with LAL waveforms, JAX-accelerated computation, and artificial neural networks.
It handles various detector configurations, waveform approximants, and spin scenarios.

Key Features:
- Fast optimal SNR calculation using interpolation
- Inner product methods with LAL and Ripple waveform generators
- JAX and MLX acceleration for GPU/vectorized computation
- ANN-based detection probability estimation
- Support for aligned and precessing spin systems
- Probability of detection calculations with various statistical models
- Detector horizon distance estimation: analytical and numerical methods

Copyright (C) 2025 Hemantakumar Phurailatpam and Otto Hannuksela.
Distributed under MIT License.
"""

import shutil
import os
import zipfile
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

from ..utils import (  # noqa: E402
    dealing_with_psds,
    interpolator_check,
    load_json,
    save_json,
    load_ann_h5_from_module,
    load_pickle,
    load_pickle_from_module,
    load_ann_h5,
    load_json_from_module,
    get_gw_parameters,
)  # from gwsnr/utils/utils.py
from ..utils import (
    # noise_weighted_inner_prod_h_inner_h,
    noise_weighted_inner_prod_h_inner_h_slim,
    _init_worker_h_inner_h,
)  # , noise_weighted_inner_prod_d_inner_h

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
    Calculate SNR and detection probability for gravitational wave signals from compact binaries.

    Provides multiple computational methods for optimal SNR calculation:\n
    - Interpolation: Fast calculation using precomputed coefficients\n
    - Inner product: Direct computation with LAL/Ripple waveforms\n
    - JAX/MLX: GPU-accelerated computation\n
    - ANN: Neural network-based estimation\n

    Other features include:\n
    - observed SNR based Pdet calculation with various statistical models\n
    - Horizon distance estimation for detectors and detector networks\n

    Parameters
    ----------
    npool : `int`
        Number of processors for parallel computation.\n
        default: 4
    mtot_min : `float`
        Minimum total mass (solar masses) for interpolation grid.\n
        default: 9.96
    mtot_max : `float`
        Maximum total mass (solar masses). Auto-adjusted if mtot_cut=True.\n
        default: 235.0
    ratio_min : `float`
        Minimum mass ratio (m2/m1) for interpolation.\n
        default: 0.1
    ratio_max : `float`
        Maximum mass ratio for interpolation.\n
        default: 1.0
    spin_max : `float`
        Maximum aligned spin magnitude.\n
        default: 0.99
    mtot_resolution : `int`
        Grid points for total mass interpolation.\n
        default: 200
    ratio_resolution : `int`
        Grid points for mass ratio interpolation.\n
        default: 20
    spin_resolution : `int`
        Grid points for spin interpolation (aligned-spin methods).\n
        default: 10
    batch_size_interpolation : `int`
        Batch size for interpolation calculations.\n
        default: 1000000
    sampling_frequency : `float`
        Detector sampling frequency (Hz).\n
        default: 2048.0
    waveform_approximant : `str`
        Bilby waveform model: 'IMRPhenomD', 'IMRPhenomXPHM', 'TaylorF2', etc.\n
        default: 'IMRPhenomD'
    frequency_domain_source_model : `str`
        Bilby frequency domain source model function for waveform generation.\n
        default: 'lal_binary_black_hole'
    minimum_frequency : `float`
        Minimum frequency (Hz) for waveform generation.\n
        default: 20.0
    reference_frequency : `float`
        Reference frequency (Hz). Optional.\n
        default: minimum_frequency.
    duration_max : `float`
        Maximum waveform duration (seconds). Optional. Auto-set for some approximants.
    duration_min : `float`
        Minimum waveform duration (seconds). Optional.
    fixed_duration : `float`
        Fixed duration (seconds) for all waveforms. Optional.
    mtot_cut : `bool`
        If True, limit mtot_max based on minimum_frequency.\n
        default: False
    snr_method : `str`
        SNR calculation method. Options:\n
        - 'interpolation_no_spins[_numba/_jax/_mlx]'\n
        - 'interpolation_aligned_spins[_numba/_jax/_mlx]'\n
        - 'inner_product[_jax]'\n
        - 'ann'\n
        default : 'interpolation_aligned_spins'
    snr_type : `str`
        SNR type: 'optimal_snr' or 'observed_snr' (not implemented).\n
        default: 'optimal_snr'
    noise_realization : `numpy.ndarray`
        Noise realization for observed SNR (not implemented). Optional.
    psds : `dict`
        Detector power spectral densities. Optional.\n
        Options:\n
        - None: Use bilby defaults\n
        - {'H1': 'aLIGODesign', 'L1': 'aLIGODesign'}: Use bilby default PSD names\n
        - {'H1': 'custom_psd.txt'} or {'H1': 'custom_asd.txt'}: Use custom PSD/ASD files. File should contain two columns: frequency and PSD/ASD values.\n
        - {'H1': 1234567890}: Use GPS time for data-based PSD\n
    ifos : `list`
        Custom interferometer objects. Defaults from psds if None. Optional.\n
        Options:\n
        - None: Use bilby defaults\n
        - ['H1', 'L1']: Uses bilby default detector configuration\n
        - Custom ifos and psds example:
        >>> import bilby
        >>> from gwsnr import GWSNR
        >>> ifosLIO = bilby.gw.detector.interferometer.Interferometer(
                name = 'LIO',
                power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file='your_asd.txt'),
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
        Directory for storing interpolation coefficients. Optional.
    create_new_interpolator : `bool`
        If True, regenerate interpolation coefficients. Optional.
    gwsnr_verbose : `bool`
        Print initialization parameters.
    multiprocessing_verbose : `bool`
        Show progress bars during computation.\n
        default: True
    pdet_kwargs : `dict`
        Detection probability parameters.
        Default: {'snr_th': 10.0, 'snr_th_net': 10.0, 'pdet_type': 'boolean', 'distribution_type': 'gaussian'}
    ann_path_dict : `dict` or `str`
        Paths to ANN models. Uses built-in models if None. Optional.
    snr_recalculation : `bool`
        Enable hybrid recalculation near detection threshold. Optional.
    snr_recalculation_range : `list`, default=[6,14]
        SNR range [min, max] for triggering recalculation. Optional.
    snr_recalculation_waveform_approximant : `str`
        Waveform approximant for recalculation. Optional.
        Default: 'IMRcd gwsnrPhenomXPHM'

    Examples
    --------
    Basic interpolation usage:

    >>> from gwsnr import GWSNR
    >>> gwsnr = GWSNR()
    >>> snrs = gwsnr.optimal_snr(mass_1=30, mass_2=30, luminosity_distance=1000, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
    >>> pdet = gwsnr.pdet(mass_1=30, mass_2=30, luminosity_distance=1000, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
    >>> print(f"SNR value: {snrs}")
    >>> print(f"P_det value: {pdet}")


    Instance Methods
    ----------
    GWSNR class has the following methods: \n
    +------------------------------------------------+------------------------------------------------+
    | Method                                         | Description                                    |
    +================================================+================================================+
    | :meth:`~calculate_mtot_max`                    | Calculate maximum total mass cutoff based on   |
    |                                                | minimum frequency                              |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~optimal_snr`                           | Primary interface for SNR calculation          |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~optimal_snr_with_ann`                  | Calculate SNR using artificial neural network  |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~optimal_snr_with_interpolation`        | Calculate SNR using bicubic interpolation      |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~optimal_snr_with_inner_product`        | Calculate SNR using LAL waveforms and inner    |
    |                                                | products                                       |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~optimal_snr_with_inner_product_ripple` | Calculate SNR using JAX-accelerated Ripple     |
    |                                                | waveforms                                      |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~pdet`                                  | Calculate probability of detection             |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~horizon_distance_analytical`           | Calculate detector horizon using analytical    |
    |                                                | formula                                        |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~horizon_distance_numerical`            | Calculate detector horizon with optimal sky    |
    |                                                | positioning                                    |
    +------------------------------------------------+------------------------------------------------+

    Instance Attributes
    ----------
    GWSNR class has the following attributes: \n
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | Attribute                                      | Type             | Unit  | Description                                    |
    +================================================+==================+=======+================================================+
    | :meth:`~npool`                                 | ``int``          |       | Number of processors for parallel processing   |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~mtot_min`                              | ``float``        | M☉    | Minimum total mass for interpolation grid      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~mtot_max`                              | ``float``        | M☉    | Maximum total mass for interpolation grid      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~ratio_min`                             | ``float``        |       | Minimum mass ratio (q = m2/m1)                 |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~ratio_max`                             | ``float``        |       | Maximum mass ratio                             |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~spin_max`                              | ``float``        |       | Maximum aligned spin magnitude                 |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~mtot_resolution`                       | ``int``          |       | Grid resolution for total mass interpolation   |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~ratio_resolution`                      | ``int``          |       | Grid resolution for mass ratio interpolation   |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~spin_resolution`                       | ``int``          |       | Grid resolution for aligned spin interpolation |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~ratio_arr`                             | ``ndarray``      |       | Mass ratio interpolation grid points           |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~mtot_arr`                              | ``ndarray``      | M☉    | Total mass interpolation grid points           |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~a_1_arr`                               | ``ndarray``      |       | Primary aligned spin interpolation grid        |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~a_2_arr`                               | ``ndarray``      |       | Secondary aligned spin interpolation grid      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~sampling_frequency`                    | ``float``        | Hz    | Detector sampling frequency                    |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~waveform_approximant`                  | ``str``          |       | LAL waveform approximant name                  |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~frequency_domain_source_model`         | ``str``          |       | Bilby frequency domain source model function   |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~f_min`                                 | ``float``        | Hz    | Minimum waveform frequency                     |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~f_ref`                                 | ``float``        | Hz    | Reference frequency for waveform generation    |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~duration_max`                          | ``float/None``   | s     | Maximum waveform duration                      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~duration_min`                          | ``float/None``   | s     | Minimum waveform duration                      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_method`                            | ``str``          |       | SNR calculation method                         |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_type`                              | ``str``          |       | SNR type: 'optimal_snr' or 'observed_snr'      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~noise_realization`                     | ``ndarray/None`` |       | Noise realization for observed SNR             |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~psds_list`                             | ``list``         |       | Detector power spectral densities              |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~detector_tensor_list`                  | ``list``         |       | Detector tensors for antenna response          |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~detector_list`                         | ``list``         |       | Detector names (e.g., ['H1', 'L1', 'V1'])      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~ifos`                                  | ``list``         |       | Bilby interferometer objects                   |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~interpolator_dir`                      | ``str``          |       | Directory for interpolation coefficients       |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~path_interpolator`                     | ``list``         |       | Paths to interpolation coefficient files       |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_partialsacaled_list`               | ``list``         |       | Partial-scaled SNR interpolation coefficients  |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~multiprocessing_verbose`               | ``bool``         |       | Show progress bars for computations            |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~identifier_dict`                       | ``dict``         |       | Interpolator parameter dictionary for caching  |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_th`                                | ``float``        |       | Individual detector SNR threshold              |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_th_net`                            | ``float``        |       | Network SNR threshold                          |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~model_dict`                            | ``dict``         |       | ANN models for each detector                   |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~scaler_dict`                           | ``dict``         |       | ANN feature scalers for each detector          |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~error_adjustment`                      | ``dict``         |       | ANN error correction parameters                |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~ann_catalogue`                         | ``dict``         |       | ANN model configuration and paths              |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_recalculation`                     | ``bool``         |       | Enable hybrid SNR recalculation                |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_recalculation_range`               | ``list``         |       | SNR range [min, max] triggering recalculation  |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~snr_recalculation_waveform_approximant`| ``str``          |       | Waveform approximant for recalculation         |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~get_interpolated_snr`                  | ``function``     |       | Interpolated SNR calculation function          |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~noise_weighted_inner_product_jax`      | ``function``     |       | JAX-accelerated inner product function         |
    +------------------------------------------------+------------------+-------+------------------------------------------------+

    Notes
    -----
    - Interpolation methods: fastest for population studies
    - Inner product methods: most accurate for individual events
    - JAX/MLX methods: leverage GPU acceleration
    - ANN methods: fast detection probability, lower SNR accuracy
    """

    def __init__(
        self,
        # General settings
        npool=int(4),
        snr_method="interpolation_aligned_spins",
        snr_type="optimal_snr",
        gwsnr_verbose=True,
        multiprocessing_verbose=True,
        pdet_kwargs=None,
        # Settings for interpolation grid
        mtot_min=2
        * 4.98,  # 4.98 Mo is the minimum component mass of BBH systems in GWTC-3
        mtot_max=2 * 112.5
        + 10.0,  # 112.5 Mo is the maximum component mass of BBH systems in GWTC-3. 10.0 Mo is added to avoid edge effects.
        ratio_min=0.1,
        ratio_max=1.0,
        spin_max=0.99,
        mtot_resolution=200,
        ratio_resolution=20,
        spin_resolution=10,
        batch_size_interpolation=1000000,
        interpolator_dir="./interpolator_json",
        create_new_interpolator=False,
        # GW signal settings
        sampling_frequency=2048.0,
        waveform_approximant="IMRPhenomD",
        frequency_domain_source_model="lal_binary_black_hole",
        minimum_frequency=20.0,
        reference_frequency=None,
        duration_max=None,
        duration_min=None,
        fixed_duration=None,
        mtot_cut=False,
        # Detector settings
        psds=None,
        ifos=None,
        noise_realization=None,  # not implemented yet
        # ANN settings
        ann_path_dict=None,
        # Hybrid SNR recalculation settings
        snr_recalculation=False,
        snr_recalculation_range=[6, 14],
        snr_recalculation_waveform_approximant="IMRPhenomXPHM",
    ):

        # getting interpolator data from the package
        # first check if the interpolator directory './interpolator_json' exists
        if not pathlib.Path(interpolator_dir).exists():
            # Get the path to the zip resource
            with path("gwsnr.core.core_data", "interpolator_json.zip") as zip_path:
                print(
                    f"Extracting interpolator data from {zip_path} to the current working directory."
                )
                zip_path = pathlib.Path(zip_path)  # Ensure it's a Path object

                # Define destination path (current working directory)
                dest_path = pathlib.Path.cwd()

                # Extract the zip file, skipping __MACOSX metadata
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        # Skip __MACOSX directory and its contents
                        if member.startswith("__MACOSX"):
                            continue
                        zip_ref.extract(member, dest_path)

        print("\nInitializing GWSNR class...\n")
        # setting instance attributes
        self.npool = npool
        self.pdet_kwargs = (
            pdet_kwargs
            if pdet_kwargs is not None
            else dict(
                snr_th=10.0,
                snr_th_net=10.0,
                pdet_type="boolean",
                distribution_type="noncentral_chi2",
                include_optimal_snr=False,
                include_observed_snr=False,
            )
        )

        self.duration_max = duration_max
        self.duration_min = duration_min
        self.fixed_duration = fixed_duration
        self.snr_method = snr_method
        self.snr_type = snr_type

        if self.snr_method == "observed_snr":
            raise ValueError(
                "'observed_snr' not implemented yet. Use 'optimal_snr' instead."
            )

        self.noise_realization = noise_realization
        self.spin_max = spin_max
        self.batch_size_interpolation = batch_size_interpolation

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
            self.snr_recalculation_waveform_approximant = (
                snr_recalculation_waveform_approximant
            )

        self.ratio_arr = np.geomspace(ratio_min, ratio_max, ratio_resolution)
        self.mtot_arr = np.sort(
            mtot_min + mtot_max - np.geomspace(mtot_min, mtot_max, mtot_resolution)
        )
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
        psds_list, detector_tensor_list, detector_list, self.ifos = dealing_with_psds(
            psds, ifos, minimum_frequency, sampling_frequency
        )

        # identifier_dict is an identifier for the interpolator
        self.identifier_dict = {
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
        if waveform_approximant == "IMRPhenomXPHM" and duration_max is None:
            print(
                "Intel processor has trouble allocating memory when the data is huge. So, by default for IMRPhenomXPHM, duration_max = 64.0. Otherwise, set to some max value like duration_max = 600.0 (10 mins)"
            )
            self.duration_max = 64.0
            self.duration_min = 4.0

        # now generate interpolator, if not exists
        list_no_spins = [
            "interpolation",
            "interpolation_no_spins",
            "interpolation_no_spins_numba",
            "interpolation_no_spins_jax",
            "interpolation_no_spins_mlx",
        ]
        list_aligned_spins = [
            "interpolation_aligned_spins",
            "interpolation_aligned_spins_numba",
            "interpolation_aligned_spins_jax",
            "interpolation_aligned_spins_mlx",
        ]

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
            self.path_interpolator = self._interpolator_setup(
                interpolator_dir,
                create_new_interpolator,
                psds_list,
                detector_tensor_list,
                detector_list,
            )

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

            self.identifier_dict["spin_max"] = self.spin_max
            self.identifier_dict["spin_resolution"] = self.spin_resolution
            # dealing with interpolator
            self.path_interpolator = self._interpolator_setup(
                interpolator_dir,
                create_new_interpolator,
                psds_list,
                detector_tensor_list,
                detector_list,
            )

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
                reference_frequency=(
                    self.f_ref if self.f_ref is not None else self.f_min
                ),
            )

            self.noise_weighted_inner_product_jax = (
                ripple_class.noise_weighted_inner_product_jax
            )

        # ANN method still needs the partialscaledSNR interpolator.
        elif snr_method == "ann":

            from ..numba import get_interpolated_snr_aligned_spins_numba

            self.get_interpolated_snr = get_interpolated_snr_aligned_spins_numba
            # below is added to find the genereated interpolator path
            self.identifier_dict["spin_max"] = self.spin_max
            self.identifier_dict["spin_resolution"] = self.spin_resolution

            (
                self.model_dict,
                self.scaler_dict,
                self.error_adjustment,
                self.ann_catalogue,
            ) = self._ann_initilization(
                ann_path_dict,
                detector_list,
                sampling_frequency,
                minimum_frequency,
                waveform_approximant,
            )
            # dealing with interpolator
            self.snr_method = "interpolation_aligned_spins"
            self.path_interpolator = self._interpolator_setup(
                interpolator_dir,
                create_new_interpolator,
                psds_list,
                detector_tensor_list,
                detector_list,
            )
            self.snr_method = "ann"

        else:
            raise ValueError(
                "SNR function type not recognised. Please choose from 'interpolation', 'interpolation_no_spins', 'interpolation_no_spins_numba', 'interpolation_no_spins_jax', 'interpolation_no_spins_mlx', 'interpolation_aligned_spins', 'interpolation_aligned_spins_numba', 'interpolation_aligned_spins_jax', 'interpolation_aligned_spins_mlx', 'inner_product', 'inner_product_jax', 'ann'."
            )

        # change back to original
        self.psds_list = psds_list
        self.detector_tensor_list = detector_tensor_list
        self.detector_list = detector_list

        if (snr_method == "inner_product") or (snr_method == "inner_product_jax"):
            self.optimal_snr_with_interpolation = self._print_no_interpolator

        # print some info
        self._print_all_params(gwsnr_verbose)
        print("\n")

    # dealing with interpolator
    def _interpolator_setup(
        self,
        interpolator_dir,
        create_new_interpolator,
        psds_list,
        detector_tensor_list,
        detector_list,
    ):
        """
        Set up interpolator files for fast SNR calculation using precomputed coefficients.

        This method manages the creation and loading of partialscaled SNR interpolation data.
        It checks for existing interpolators, generates missing ones, and loads coefficients
        for runtime use.

        Parameters
        ----------
        interpolator_dir : str
            Directory path for storing interpolator json files.
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
            File paths to interpolator json files for all detectors.

        Notes
        -----
        - Uses :func:`interpolator_check` to identify missing interpolators
        - Calls :meth:`_init_partialscaled` to generate new coefficients
        - Loads coefficients into :meth:`snr_partialsacaled_list` for runtime use
        """

        # Note: it will only select detectors that does not have interpolator stored yet
        (
            self.psds_list,
            self.detector_tensor_list,
            self.detector_list,
            self.path_interpolator,
            path_interpolator_all,
        ) = interpolator_check(
            identifier_dict=self.identifier_dict.copy(),
            interpolator_dir=interpolator_dir,
            create_new=create_new_interpolator,
        )

        # len(detector_list) == 0, means all the detectors have interpolator stored
        if len(self.detector_list) > 0:
            print("Please be patient while the interpolator is generated")
            # if self.snr_method == 'interpolation_aligned_spins':
            #     self._init_partialscaled_aligned_spins()
            # else:
            self._init_partialscaled()
        elif create_new_interpolator:
            # change back to original
            self.psds_list = psds_list
            self.detector_tensor_list = detector_tensor_list
            self.detector_list = detector_list
            print("Please be patient while the interpolator is generated")
            # if self.snr_method == 'interpolation_aligned_spins':
            #     self._init_partialscaled_aligned_spins()
            # else:
            self._init_partialscaled()

        # get all partialscaledSNR from the stored interpolator
        self.snr_partialsacaled_list = np.array(
            [load_json(path) for path in path_interpolator_all], dtype=np.float64
        )

        return path_interpolator_all

    def _ann_initilization(
        self,
        ann_path_dict,
        detector_list,
        sampling_frequency,
        minimum_frequency,
        waveform_approximant,
    ):
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
        if not pathlib.Path("./ann_data").exists():
            # Get the path to the resource
            with path("gwsnr.ann", "ann_data") as resource_path:
                print(
                    f"Copying ANN data from the library resource {resource_path} to the current working directory."
                )
                resource_path = pathlib.Path(resource_path)  # Ensure it's a Path object

                # Define destination path (same name in current working directory)
                dest_path = pathlib.Path.cwd() / resource_path.name

                # Copy entire directory tree
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(resource_path, dest_path)

        if ann_path_dict is None:
            print("ANN model and scaler path is not given. Using the default path.")
            ann_path_dict = "./ann_data/ann_path_dict.json"
        else:
            print("ANN model and scaler path is given. Using the given path.")

        if isinstance(ann_path_dict, str):
            ann_path_dict = load_json(ann_path_dict)
        elif isinstance(ann_path_dict, dict):
            pass
        else:
            raise ValueError(
                "ann_path_dict should be a dictionary or a path to the json file."
            )

        model_dict = {}
        scaler_dict = {}
        error_adjustment = {}
        # loop through the detectors
        for detector in detector_list:
            if detector not in ann_path_dict.keys():
                # check if the model and scaler is available
                raise ValueError(
                    f"ANN model and scaler for {detector} is not available. Please provide the path to the model and scaler. Refer to the 'gwsnr' documentation for more information on how to add new ANN model."
                )
            else:
                # check of model parameters
                check = True
                check &= (
                    sampling_frequency == ann_path_dict[detector]["sampling_frequency"]
                )
                check &= (
                    minimum_frequency == ann_path_dict[detector]["minimum_frequency"]
                )
                check &= (
                    waveform_approximant
                    == ann_path_dict[detector]["waveform_approximant"]
                )
                # check &= (snr_th == ann_path_dict[detector]['snr_th']) # this has been deprecated
                # check for the model and scaler keys exit or not
                check &= "model_path" in ann_path_dict[detector].keys()
                check &= "scaler_path" in ann_path_dict[detector].keys()

                if not check:
                    raise ValueError(
                        f"ANN model parameters for {detector} is not suitable for the given gwsnr parameters. Existing parameters are: {ann_path_dict[detector]}"
                    )

            # get ann model
            if not os.path.exists(ann_path_dict[detector]["model_path"]):
                # load the model from gwsnr/ann/data directory
                model_dict[detector] = load_ann_h5_from_module(
                    "gwsnr", "ann.data", ann_path_dict[detector]["model_path"]
                )
                print(
                    f"ANN model for {detector} is loaded from gwsnr/ann/data directory."
                )
            else:
                # load the model from the given path
                model_dict[detector] = load_ann_h5(
                    ann_path_dict[detector]["model_path"]
                )
                print(
                    f"ANN model for {detector} is loaded from {ann_path_dict[detector]['model_path']}."
                )

            # get ann scaler
            if not os.path.exists(ann_path_dict[detector]["scaler_path"]):
                # load the scaler from gwsnr/ann/data directory
                scaler_dict[detector] = load_pickle_from_module(
                    "gwsnr", "ann.data", ann_path_dict[detector]["scaler_path"]
                )
                print(
                    f"ANN scaler for {detector} is loaded from gwsnr/ann/data directory."
                )
            else:
                # load the scaler from the given path
                scaler_dict[detector] = load_pickle(
                    ann_path_dict[detector]["scaler_path"]
                )
                print(
                    f"ANN scaler for {detector} is loaded from {ann_path_dict[detector]['scaler_path']}."
                )

            # get error_adjustment
            if not os.path.exists(ann_path_dict[detector]["error_adjustment_path"]):
                # load the error_adjustment from gwsnr/ann/data directory
                error_adjustment[detector] = load_json_from_module(
                    "gwsnr",
                    "ann.data",
                    ann_path_dict[detector]["error_adjustment_path"],
                )
                print(
                    f"ANN error_adjustment for {detector} is loaded from gwsnr/ann/data directory."
                )
            else:
                # load the error_adjustment from the given path
                error_adjustment[detector] = load_json(
                    ann_path_dict[detector]["error_adjustment_path"]
                )
                print(
                    f"ANN error_adjustment for {detector} is loaded from {ann_path_dict[detector]['error_adjustment_path']}."
                )

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
            Always raised, suggesting to use interpolation-based :meth:`~snr_method`.
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

            return 1.1 * findchirp_chirptime(mass_1, mass_2, minimum_frequency)

        # find where func is zero
        from scipy.optimize import fsolve

        mtot_max_generated = fsolve(func, 184)[
            0
        ]  # to make sure that chirptime is not negative, TaylorF2 might need this
        if mtot_max > mtot_max_generated:
            mtot_max = mtot_max_generated

        return mtot_max

    def _print_all_params(self, verbose=True):
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
            print(
                "reference frequency (f_ref): ",
                self.f_ref if self.f_ref is not None else self.f_min,
            )
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
        mass_1=np.array(
            [
                10.0,
            ]
        ),
        mass_2=np.array(
            [
                10.0,
            ]
        ),
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
        gw_param_dict=None,
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
        gw_param_dict : dict or None, default=None
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
            (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
            Values are arrays matching input size.

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
        >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")

        >>> # Multiple systems with parameter dictionary
        >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
        >>> result = snr.optimal_snr(gw_param_dict=params)
        """

        gw_param_dict = gw_param_dict.copy() if gw_param_dict else None

        if gw_param_dict is None:
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
                "eccentricity": eccentricity,
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
            gw_param_dict["a_1"] = a_1_old
            gw_param_dict["a_2"] = a_2_old

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

            snr_net = snr_dict["optimal_snr_net"]
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
        mass_1=30.0,
        mass_2=29.0,
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
        gw_param_dict=None,
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
        gw_param_dict : dict or None, default=None
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR estimates for each detector and network. Keys are 'optimal_snr_{detector}'
            (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.

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
        >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")
        """

        gw_param_dict = gw_param_dict.copy() if gw_param_dict else None

        if gw_param_dict is not None:
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
                _,
                _,
                _,
            ) = get_gw_parameters(gw_param_dict)
        else:
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
                _,
                _,
                _,
            ) = get_gw_parameters(
                dict(
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
            )

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
        ann_input = self._output_ann(idx2, params)

        # 1. load the model 2. load feature scaler 3. predict snr
        optimal_snr = {f"optimal_snr_{det}": np.zeros(size) for det in detectors}
        optimal_snr["optimal_snr_net"] = np.zeros(size)
        for i, det in enumerate(detectors):
            x = scaler[det].transform(ann_input[i])
            optimal_snr_ = model[det].predict(x, verbose=0).flatten()
            # adjusting the optimal SNR with error adjustment
            optimal_snr[f"optimal_snr_{det}"][idx_tracker] = optimal_snr_ - (
                self.error_adjustment[det]["slope"] * optimal_snr_
                + self.error_adjustment[det]["intercept"]
            )
            optimal_snr["optimal_snr_net"] += optimal_snr[f"optimal_snr_{det}"] ** 2
        optimal_snr["optimal_snr_net"] = np.sqrt(optimal_snr["optimal_snr_net"])

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json(output_filename, optimal_snr)

        return optimal_snr

    def _output_ann(self, idx, params):
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

        mass_1 = np.array(params["mass_1"][idx])
        mass_2 = np.array(params["mass_2"][idx])
        luminosity_distance = np.array(params["luminosity_distance"][idx])
        theta_jn = np.array(params["theta_jn"][idx])
        psi = np.array(params["psi"][idx])
        geocent_time = np.array(params["geocent_time"][idx])
        ra = np.array(params["ra"][idx])
        dec = np.array(params["dec"][idx])
        a_1 = np.array(params["a_1"][idx])
        a_2 = np.array(params["a_2"][idx])
        tilt_1 = np.array(params["tilt_1"][idx])
        tilt_2 = np.array(params["tilt_2"][idx])
        # effective spin
        chi_eff = (mass_1 * a_1 * np.cos(tilt_1) + mass_2 * a_2 * np.cos(tilt_2)) / (
            mass_1 + mass_2
        )

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
        eta = mass_1 * mass_2 / (mass_1 + mass_2) ** 2.0
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

        return ann_input

    def optimal_snr_with_interpolation(
        self,
        mass_1=30.0,
        mass_2=29.0,
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
        gw_param_dict=None,
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
        gw_param_dict : dict or None, default=None
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
            (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
            Systems outside mass bounds have zero SNR.

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
        >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")
        """

        # getting the parameters from the dictionary
        gw_param_dict = gw_param_dict.copy() if gw_param_dict else None

        if gw_param_dict is not None:
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
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = get_gw_parameters(gw_param_dict)
        else:
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
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = get_gw_parameters(
                dict(
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
                )
            )

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
        optimal_snr = {f"optimal_snr_{det}": np.zeros(size) for det in detectors}
        optimal_snr["optimal_snr_net"] = np.zeros(size)
        for j, det in enumerate(detectors):
            optimal_snr[f"optimal_snr_{det}"][idx_tracker] = snr[j]
        optimal_snr["optimal_snr_net"][idx_tracker] = snr_effective

        # Save as JSON file
        if output_jsonfile:
            output_filename = (
                output_jsonfile if isinstance(output_jsonfile, str) else "snr.json"
            )
            save_json(output_filename, optimal_snr)

        return optimal_snr

    def _init_partialscaled(self):
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

        Coefficients are saved as json files for runtime interpolation.

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

        list_no_spins = [
            "interpolation",
            "interpolation_no_spins",
            "interpolation_no_spins_numba",
            "interpolation_no_spins_jax",
            "interpolation_no_spins_mlx",
        ]
        list_aligned_spins = [
            "interpolation_aligned_spins",
            "interpolation_aligned_spins_numba",
            "interpolation_aligned_spins_jax",
            "interpolation_aligned_spins_mlx",
        ]

        # Create broadcastable 4D grids
        if self.snr_method in list_aligned_spins:
            a_1_table = self.a_1_arr.copy()
            a_2_table = self.a_2_arr.copy()
            size3 = self.spin_resolution
            size4 = self.spin_resolution
            a_1_table = np.asarray(a_1_table)
            a_2_table = np.asarray(a_2_table)

            q, mtot, a_1, a_2 = np.meshgrid(
                ratio_table, mtot_table, a_1_table, a_2_table, indexing="ij"
            )
        elif self.snr_method in list_no_spins:
            q, mtot = np.meshgrid(ratio_table, mtot_table, indexing="ij")
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
                antenna_response_cross(
                    ra_,
                    dec_,
                    geocent_time_,
                    psi_,
                    tensor,
                )
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
        Mchirp = ((mass_1 * mass_2) ** (3 / 5)) / (
            (mass_1 + mass_2) ** (1 / 5)
        )  # shape (size1, size2, size3, size4)
        Mchirp_scaled = Mchirp ** (5.0 / 6.0)
        # filling in interpolation table for different detectors

        for j in num_det:
            if self.snr_method in list_aligned_spins:
                snr_partial_ = (
                    np.array(
                        np.reshape(
                            optimal_snr_unscaled[f"optimal_snr_{detectors[j]}"],
                            (size1, size2, size3, size4),
                        )
                        * dl_eff[j]
                        / Mchirp_scaled,
                        dtype=np.float32,
                    ),
                )  # shape (size1, size2, size3, size4)
            elif self.snr_method in list_no_spins:
                snr_partial_ = (
                    np.array(
                        np.reshape(
                            optimal_snr_unscaled[f"optimal_snr_{detectors[j]}"],
                            (size1, size2),
                        )
                        * dl_eff[j]
                        / Mchirp_scaled,
                        dtype=np.float32,
                    ),
                )  # shape (size1, size2, size3, size4)
            else:
                raise ValueError(
                    f"snr_method {self.snr_method} is not supported for interpolation."
                )
            # print('dl_eff=',dl_eff[j])
            # print('Mchirp_scaled=',Mchirp_scaled.shape)
            # print('optimal_snr_unscaled=',np.reshape(optimal_snr_unscaled[detectors[j]],(size1, size2, size3, size4)).shape)
            print(
                f"\nSaving Partial-SNR for {detectors[j]} detector with shape {snr_partial_[0].shape}"
            )
            save_json(self.path_interpolator[j], snr_partial_[0])

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
        gw_param_dict=None,
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
        gw_param_dict : dict or None, default=None
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
            (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
            Systems outside mass bounds have zero SNR.

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
        >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")
        """

        # if gw_param_dict is given, then use that
        gw_param_dict = gw_param_dict.copy() if gw_param_dict else None

        if gw_param_dict is not None:
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
                lambda_1,
                lambda_2,
                eccentricity,
            ) = get_gw_parameters(gw_param_dict)
        else:
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
                lambda_1,
                lambda_2,
                eccentricity,
            ) = get_gw_parameters(
                dict(
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
                )
            )

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
            approx_duration = safety * findchirp_chirptime(
                mass_1[idx], mass_2[idx], f_min
            )
            duration = np.ceil(approx_duration + 2.0)
            if self.duration_max:
                duration[duration > self.duration_max] = (
                    self.duration_max
                )  # IMRPheonomXPHM has maximum duration of 371s
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

        # Set up input arguments for multiprocessing (slim version - only per-item data)
        # Shared data (psd_list, approximant, etc.) is passed via Pool initializer
        # This reduces pickling overhead from N times (per work item) to M times (per worker)
        input_arguments = [
            (
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
                duration_i,
                iterations_i,
            )
            for (
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
                duration_i,
                iterations_i,
            ) in zip(
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

        if self.snr_type == "optimal_snr":

            self._multiprocessing_error()
            # Use Pool initializer to send shared data once per worker instead of per work item
            # This dramatically reduces overhead when psd_list is large
            with mp.Pool(
                processes=npool,
                initializer=_init_worker_h_inner_h,
                initargs=(
                    psd_list,
                    approximant,
                    f_min,
                    f_ref,
                    sampling_frequency,
                    frequency_domain_source_model,
                ),
            ) as pool:
                # call the same function with different data in parallel
                # imap->retain order in the list, while map->doesn't
                if self.multiprocessing_verbose:
                    for result in tqdm(
                        pool.imap_unordered(
                            noise_weighted_inner_prod_h_inner_h_slim, input_arguments
                        ),
                        total=len(input_arguments),
                        ncols=100,
                    ):
                        # but, np.shape(hp_inner_hp_i) = (size1, len(num_det))
                        hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                        hp_inner_hp[:, iter_i] = hp_inner_hp_i
                        hc_inner_hc[:, iter_i] = hc_inner_hc_i
                else:
                    # with map, without tqdm
                    for result in pool.map(
                        noise_weighted_inner_prod_h_inner_h_slim, input_arguments
                    ):
                        hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                        hp_inner_hp[:, iter_i] = hp_inner_hp_i
                        hc_inner_hc[:, iter_i] = hc_inner_hc_i

            # combining the results
            snrs_sq = abs((Fp**2) * hp_inner_hp + (Fc**2) * hc_inner_hc)
            snr = np.sqrt(snrs_sq)

        elif self.snr_type == "observed_snr":

            raise ValueError("observed_snr not implemented yet")

        else:
            raise ValueError(
                "snr_type should be either 'optimal_snr' or 'observed_snr'"
            )

        snr_effective = np.sqrt(np.sum(snr**2, axis=0))

        # organizing the snr dictionary
        optimal_snr = dict()
        for j, det in enumerate(detectors):
            snr_buffer = np.zeros(num)
            snr_buffer[idx] = snr[j]
            optimal_snr[f"optimal_snr_{det}"] = snr_buffer
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

    def _multiprocessing_error(self):
        """
        Check multiprocessing guard and raise error if not in main process.

        Raises
        ------
        RuntimeError
            If called from a subprocess instead of the main process, indicating
            the code is not properly wrapped in ``if __name__ == "__main__":``.
        """
        # to access multi-cores instead of multithreading
        if mp.current_process().name != "MainProcess":
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
        gw_param_dict=None,
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
        gw_param_dict : dict or None, default=None
            Parameter dictionary. If provided, overrides individual arguments.
        output_jsonfile : str or bool, default=False
            Save results to JSON file. If True, saves as 'snr.json'.

        Returns
        -------
        dict
            SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
            (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
            Systems outside mass bounds have zero SNR.

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
        >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")
        """

        # if gw_param_dict is given, then use that
        gw_param_dict = gw_param_dict.copy() if gw_param_dict else None

        if gw_param_dict is not None:
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
                lambda_1,
                lambda_2,
                eccentricity,
            ) = get_gw_parameters(gw_param_dict)
        else:
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
                lambda_1,
                lambda_2,
                eccentricity,
            ) = get_gw_parameters(
                dict(
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
                )
            )

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
            multiprocessing_verbose=self.multiprocessing_verbose,
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
            optimal_snr[f"optimal_snr_{det}"] = snr_buffer

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

    def pdet(
        self,
        mass_1=np.array(
            [
                10.0,
            ]
        ),
        mass_2=np.array(
            [
                10.0,
            ]
        ),
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
        gw_param_dict=None,
        output_jsonfile=False,
        snr_th=None,
        snr_th_net=None,
        pdet_type=None,
        distribution_type=None,
        include_optimal_snr=False,
        include_observed_snr=False,
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
        gw_param_dict : dict or None, default=None
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
            - 'fixed_snr': Deterministic detection based on optimal SNR (only for 'boolean' pdet_type)

        Returns
        -------
        dict
            Detection probabilities for each detector and network. Keys are 'pdet_{detector}'
            (e.g., 'pdet_H1', 'pdet_L1', 'pdet_V1') and 'pdet_net'. Values depend on pdet_type:
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
        distribution_type = (
            distribution_type
            if distribution_type
            else self.pdet_kwargs["distribution_type"]
        )
        include_optimal_snr = (
            include_optimal_snr
            if include_optimal_snr
            else self.pdet_kwargs["include_optimal_snr"]
        )
        include_observed_snr = (
            include_observed_snr
            if include_observed_snr
            else self.pdet_kwargs["include_observed_snr"]
        )

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
                    nc_param = (
                        snr_dict[f"optimal_snr_{det}"] ** 2
                    )  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                    # sum up the probabilities from snr_th to inf
                    pdet_dict[f"pdet_{det}"] = 1 - ncx2.cdf(
                        snr_th[i] ** 2, df=df, nc=nc_param
                    )

                elif distribution_type == "gaussian":
                    pdet_dict[f"pdet_{det}"] = 1 - norm.cdf(
                        snr_th[i] - snr_dict[f"optimal_snr_{det}"]
                    )

            elif pdet_type == "boolean":
                if distribution_type == "noncentral_chi2":

                    df = 2  # 2 quadratures per IFO
                    nc_param = (
                        snr_dict[f"optimal_snr_{det}"] ** 2
                    )  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                    # sum up the probabilities from snr_th to inf
                    observed_snr = np.sqrt(
                        ncx2.rvs(
                            df=df,
                            nc=nc_param,
                            size=snr_dict[f"optimal_snr_{det}"].shape,
                        )
                    )

                    pdet_dict[f"pdet_{det}"] = np.array(
                        snr_th[i] < observed_snr, dtype=int
                    )

                elif distribution_type == "gaussian":

                    observed_snr = snr_dict[f"optimal_snr_{det}"] + np.random.normal(
                        0, 1, size=snr_dict[f"optimal_snr_{det}"].shape
                    )
                    pdet_dict[f"pdet_{det}"] = np.array(
                        snr_th[i] < observed_snr, dtype=int
                    )

                elif distribution_type == "fixed_snr":

                    observed_snr = snr_dict[f"optimal_snr_{det}"]
                    pdet_dict[f"pdet_{det}"] = np.array(
                        snr_th[i] < observed_snr, dtype=int
                    )

                if include_observed_snr:
                    pdet_dict[f"observed_snr_{det}"] = observed_snr

            else:
                raise ValueError(
                    "pdet_type should be either 'boolean' or 'probability_distribution'"
                )

            if include_optimal_snr:
                pdet_dict[f"optimal_snr_{det}"] = snr_dict[f"optimal_snr_{det}"]

        # for network
        if pdet_type == "probability_distribution":

            if distribution_type == "noncentral_chi2":
                df = 2 * len(detectors)  # 2 quadratures per IFO
                nc_param = (
                    snr_dict["optimal_snr_net"] ** 2
                )  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                # sum up the probabilities from snr_th to inf
                pdet_dict["pdet_net"] = 1 - ncx2.cdf(snr_th_net**2, df=df, nc=nc_param)
            elif distribution_type == "gaussian":
                pdet_dict["pdet_net"] = np.array(
                    1 - norm.cdf(snr_th_net - snr_dict["optimal_snr_net"])
                )

        elif pdet_type == "boolean":
            if distribution_type == "noncentral_chi2":
                df = 2 * len(detectors)  # 2 quadratures per IFO
                nc_param = (
                    snr_dict["optimal_snr_net"] ** 2
                )  # non-centrality parameter. SciPy uses lambda^2 for ncx2. Essick's lambda = rho_opt
                # sum up the probabilities from snr_th to inf
                observed_snr_net = np.sqrt(
                    ncx2.rvs(df=df, nc=nc_param, size=snr_dict["optimal_snr_net"].shape)
                )
                pdet_dict["pdet_net"] = np.array(
                    snr_th_net < observed_snr_net, dtype=int
                )
            elif distribution_type == "gaussian":
                observed_snr_net = snr_dict["optimal_snr_net"] + np.random.normal(
                    0, 1, size=snr_dict["optimal_snr_net"].shape
                )
                pdet_dict["pdet_net"] = np.array(
                    snr_th_net < observed_snr_net, dtype=int
                )
            elif distribution_type == "fixed_snr":
                observed_snr_net = snr_dict["optimal_snr_net"]
                pdet_dict["pdet_net"] = np.array(
                    snr_th_net < observed_snr_net, dtype=int
                )

            if include_observed_snr:
                pdet_dict["observed_snr_net"] = observed_snr_net

        if include_optimal_snr:
            pdet_dict["optimal_snr_net"] = snr_dict["optimal_snr_net"]

        return pdet_dict

    def horizon_distance_analytical(self, mass_1=1.4, mass_2=1.4, snr_th=None):
        """
        Calculate detector horizon distance for compact binary coalescences. Follows analytical formula from arXiv:gr-qc/0509116 .

        This method doesn't calculate horizon distance for the detector network, but for individual detectors only. Use horizon_distance_numerical for network horizon.

        Computes the maximum range at which a source can be detected with optimal orientation (face-on, overhead). Uses reference SNR at 100 Mpc scaled by  effective distance and detection threshold.

        Parameters
        ----------
        mass_1 : array_like or float, default=1.4
            Primary mass in solar masses.
        mass_2 : array_like or float, default=1.4
            Secondary mass in solar masses.
        snr_th : float, optional
            Individual detector SNR threshold. Uses class default if None.

        Returns
        -------
        horizon_distance_dict : dict
            Horizon distances in Mpc for each detector.
            Keys: 'horizon_distance_{detector}' (e.g., 'horizon_distance_H1', 'horizon_distance_L1').
            Values: horizon distance in Mpc for the corresponding detector.

        Notes
        -----
        - Assumes optimal orientation: θ_jn=0, overhead sky location
        - Formula: d_horizon = (d_eff/SNR_th) x SNR_opt
        - Compatible with all waveform approximants
        - Does not calculate network horizon; use horizon_distance_numerical for network

        Examples
        --------
        >>> snr = GWSNR(snr_method='inner_product')
        >>> horizon = snr.horizon_distance_analytical(mass_1=1.4, mass_2=1.4)
        >>> print(f"H1 horizon: {horizon['horizon_distance_H1']:.1f} Mpc")
        """

        from ..numba import effective_distance

        if snr_th:
            snr_th = snr_th
        else:
            snr_th = self.pdet_kwargs["snr_th"]

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

        horizon = {}
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
                detector_tensor=detector_tensor[j],
            )

            # Horizon calculation
            horizon[f"horizon_distance_{det}"] = (
                dl_eff[j] / snr_th
            ) * optimal_snr_unscaled[f"optimal_snr_{det}"]

        return horizon

    def horizon_distance_numerical(
        self,
        mass_1=1.4,
        mass_2=1.4,
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
        mass_1 : float, default=1.4
            Primary mass in solar masses.
        mass_2 : float, default=1.4
            Secondary mass in solar masses.
        psi : float, default=0.0
            Polarization angle in radians.
        phase : float, default=0.0
            Coalescence phase in radians.
        geocent_time : float, default=1246527224.169434
            GPS coalescence time at geocenter in seconds.
        a_1 : float, default=0.0
            Primary spin magnitude (dimensionless).
        a_2 : float, default=0.0
            Secondary spin magnitude (dimensionless).
        tilt_1 : float, default=0.0
            Primary spin tilt angle in radians.
        tilt_2 : float, default=0.0
            Secondary spin tilt angle in radians.
        phi_12 : float, default=0.0
            Azimuthal angle between spins in radians.
        phi_jl : float, default=0.0
            Azimuthal angle between total and orbital angular momentum in radians.
        lambda_1 : float, default=0.0
            Primary tidal deformability (dimensionless).
        lambda_2 : float, default=0.0
            Secondary tidal deformability (dimensionless).
        eccentricity : float, default=0.0
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
            Horizon distances in Mpc for each detector and network.
            Keys: 'horizon_distance_{detector}' (e.g., 'horizon_distance_H1', 'horizon_distance_L1')
            and 'horizon_distance_net' for the network.
        optimal_sky_location : dict
            Optimal sky coordinates (ra, dec) in radians for maximum SNR at given geocent_time.
            Keys: 'optimal_sky_location_{detector}' (e.g., 'optimal_sky_location_H1')
            and 'optimal_sky_location_net' for the network.

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
        >>> print(f"Network horizon: {horizon['horizon_distance_net']:.1f} Mpc at (RA={sky['optimal_sky_location_net'][0]:.2f}, Dec={sky['optimal_sky_location_net'][1]:.2f})")
        """

        # find root i.e. snr = snr_th
        from scipy.optimize import root_scalar
        from scipy.optimize import differential_evolution
        from astropy.time import Time
        from astropy.coordinates import SkyCoord, AltAz, EarthLocation
        import astropy.units as u

        # check: all input GW parameters must be floats (not arrays)
        for param in [
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
            lambda_1,
            lambda_2,
            eccentricity,
        ]:
            if isinstance(param, np.ndarray):
                raise TypeError(
                    "All GW input parameters must be floats for horizon_distance_numerical."
                )

        snr_th = snr_th if snr_th else self.pdet_kwargs["snr_th"]
        snr_th_net = snr_th_net if snr_th_net else self.pdet_kwargs["snr_th_net"]

        if minimize_function_dict is None:
            minimize_function_dict = dict(
                bounds=[(0, 2 * np.pi), (-np.pi / 2, np.pi / 2)],
                tol=1e-7,
                polish=True,
                maxiter=10000,
            )

        if root_scalar_dict is None:
            root_scalar_dict = dict(bracket=[1, 100000], method="bisect", xtol=1e-5)

        detectors = self.detector_list.copy()
        detectors.append("net")
        detectors = np.array(detectors)

        horizon = {}
        optimal_sky_location = {}  # ra, dec

        if detector_location_as_optimal_sky:
            ifos = self.ifos

        for i, det in enumerate(detectors):

            # minimize this function
            # do for network detectors first
            if det == "net":

                def snr_minimize(x):
                    """Minimize the inverse of SNR to maximize SNR"""
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
                    )["optimal_snr_net"][0]

                    return 1 / snr

                # Use differential evolution to find the ra and dec that maximize the antenna response
                ra_max, dec_max = differential_evolution(
                    snr_minimize,
                    bounds=minimize_function_dict["bounds"],  # Only ra and dec bounds
                    tol=minimize_function_dict["tol"],
                    polish=minimize_function_dict["polish"],
                    maxiter=minimize_function_dict["maxiter"],
                ).x

            else:
                # for individual detectors, find the sky location that maximizes (F_plus^2 + F_cross^2)
                if detector_location_as_optimal_sky:
                    # use astropy to find the zenith location of the detector at the given geocentric time
                    t = Time(geocent_time, format="gps", scale="utc")
                    loc = EarthLocation(
                        lat=ifos[i].latitude * u.deg,
                        lon=ifos[i].longitude * u.deg,
                        height=ifos[i].elevation * u.m,
                    )
                    zen = SkyCoord(
                        alt=90 * u.deg,
                        az=0 * u.deg,
                        frame=AltAz(location=loc, obstime=t),
                    ).icrs
                    ra_max = zen.ra.rad
                    dec_max = zen.dec.rad

                else:
                    # use the maximization function to find the ra and dec that maximize the antenna response
                    def antenna_response_minimization(x):
                        ra_, dec_ = x
                        detector_tensor = np.array(self.detector_tensor_list[i])

                        f_plus = antenna_response_plus(
                            ra_, dec_, geocent_time, psi, detector_tensor
                        )
                        f_cross = antenna_response_cross(
                            ra_, dec_, geocent_time, psi, detector_tensor
                        )

                        return 1 / (f_plus**2 + f_cross**2)

                    ra_max, dec_max = differential_evolution(
                        antenna_response_minimization,
                        bounds=minimize_function_dict[
                            "bounds"
                        ],  # Only ra and dec bounds
                        tol=minimize_function_dict["tol"],
                        polish=minimize_function_dict["polish"],
                        maxiter=minimize_function_dict["maxiter"],
                    ).x

                # check the maximum antenna response
                if maximization_check is True:

                    f_plus_max = antenna_response_plus(
                        ra_max, dec_max, geocent_time, psi, self.detector_tensor_list[i]
                    )
                    f_cross_max = antenna_response_cross(
                        ra_max, dec_max, geocent_time, psi, self.detector_tensor_list[i]
                    )
                    antenna_max = np.sqrt(f_plus_max**2 + f_cross_max**2)
                    # raise warning if antenna response is not close to 1
                    if not np.isclose(antenna_max, 1.0, atol=1e-2):
                        print(
                            f"\n[WARNING] Maximum antenna response for {det} is {antenna_max:.3f}, which is not close to 1.0. The horizon distance may be underestimated.\n"
                            "This could be due to the chosen geocentric time or detector configuration.\n"
                            "Consider changing the geocentric time or checking the detector tensor.\n"
                        )

            self.multiprocessing_verbose = False

            def snr_fn(dl):
                # optimal_snr_with_inner_product returns a dictionary with keys as detectors
                # and values as SNR values
                optimal_snr = self.optimal_snr(
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
                )[f"optimal_snr_{det}"][0]

                return optimal_snr - snr_th_net

            # find root i.e. snr = snr_th
            horizon[f"horizon_distance_{det}"] = root_scalar(
                snr_fn,
                bracket=root_scalar_dict["bracket"],
                method=root_scalar_dict["method"],
                xtol=root_scalar_dict["xtol"],
            ).root
            optimal_sky_location[f"optimal_sky_location_{det}"] = (ra_max, dec_max)

        return horizon, optimal_sky_location

    @property
    def npool(self):
        """
        Number of processors for parallel processing.

        Returns
        -------
        npool : `int`
            Number of processors for parallel processing.\n
            default: 4
        """
        return self._npool

    @npool.setter
    def npool(self, value):
        self._npool = value

    @property
    def mtot_min(self):
        """
        Minimum total mass for interpolation grid.

        Returns
        -------
        mtot_min : `float`
            Minimum total mass (M☉) for interpolation grid.\n
            default: 9.96
        """
        return self._mtot_min

    @mtot_min.setter
    def mtot_min(self, value):
        self._mtot_min = value

    @property
    def mtot_max(self):
        """
        Maximum total mass for interpolation grid.

        Returns
        -------
        mtot_max : `float`
            Maximum total mass (M☉) for interpolation grid.\n
            default: 235.0
        """
        return self._mtot_max

    @mtot_max.setter
    def mtot_max(self, value):
        self._mtot_max = value

    @property
    def ratio_min(self):
        """
        Minimum mass ratio for interpolation grid.

        Returns
        -------
        ratio_min : `float`
            Minimum mass ratio (q = m2/m1) for interpolation grid.\n
            default: 0.1
        """
        return self._ratio_min

    @ratio_min.setter
    def ratio_min(self, value):
        self._ratio_min = value

    @property
    def ratio_max(self):
        """
        Maximum mass ratio for interpolation grid.

        Returns
        -------
        ratio_max : `float`
            Maximum mass ratio for interpolation grid.\n
            default: 1.0
        """
        return self._ratio_max

    @ratio_max.setter
    def ratio_max(self, value):
        self._ratio_max = value

    @property
    def spin_max(self):
        """
        Maximum aligned spin magnitude for interpolation.

        Returns
        -------
        spin_max : `float`
            Maximum aligned spin magnitude for interpolation.\n
            default: 0.99
        """
        return self._spin_max

    @spin_max.setter
    def spin_max(self, value):
        self._spin_max = value

    @property
    def mtot_resolution(self):
        """
        Grid resolution for total mass interpolation.

        Returns
        -------
        mtot_resolution : `int`
            Grid resolution for total mass interpolation.\n
            default: 200
        """
        return self._mtot_resolution

    @mtot_resolution.setter
    def mtot_resolution(self, value):
        self._mtot_resolution = value

    @property
    def ratio_resolution(self):
        """
        Grid resolution for mass ratio interpolation.

        Returns
        -------
        ratio_resolution : `int`
            Grid resolution for mass ratio interpolation.\n
            default: 20
        """
        return self._ratio_resolution

    @ratio_resolution.setter
    def ratio_resolution(self, value):
        self._ratio_resolution = value

    @property
    def spin_resolution(self):
        """
        Grid resolution for aligned spin interpolation.

        Returns
        -------
        spin_resolution : `int`
            Grid resolution for aligned spin interpolation.\n
            default: 10
        """
        return self._spin_resolution

    @spin_resolution.setter
    def spin_resolution(self, value):
        self._spin_resolution = value

    @property
    def ratio_arr(self):
        """
        Mass ratio interpolation grid points.

        Returns
        -------
        ratio_arr : `numpy.ndarray`
            Mass ratio interpolation grid points.
        """
        return self._ratio_arr

    @ratio_arr.setter
    def ratio_arr(self, value):
        self._ratio_arr = value

    @property
    def mtot_arr(self):
        """
        Total mass interpolation grid points.

        Returns
        -------
        mtot_arr : `numpy.ndarray`
            Total mass (M☉) interpolation grid points.
        """
        return self._mtot_arr

    @mtot_arr.setter
    def mtot_arr(self, value):
        self._mtot_arr = value

    @property
    def a_1_arr(self):
        """
        Primary aligned spin interpolation grid.

        Returns
        -------
        a_1_arr : `numpy.ndarray`
            Primary aligned spin interpolation grid.
        """
        return self._a_1_arr

    @a_1_arr.setter
    def a_1_arr(self, value):
        self._a_1_arr = value

    @property
    def a_2_arr(self):
        """
        Secondary aligned spin interpolation grid.

        Returns
        -------
        a_2_arr : `numpy.ndarray`
            Secondary aligned spin interpolation grid.
        """
        return self._a_2_arr

    @a_2_arr.setter
    def a_2_arr(self, value):
        self._a_2_arr = value

    @property
    def sampling_frequency(self):
        """
        Detector sampling frequency.

        Returns
        -------
        sampling_frequency : `float`
            Detector sampling frequency (Hz).\n
            default: 2048.0
        """
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, value):
        self._sampling_frequency = value

    @property
    def waveform_approximant(self):
        """
        LAL waveform approximant name.

        Returns
        -------
        waveform_approximant : `str`
            LAL waveform approximant (e.g., 'IMRPhenomD', 'IMRPhenomXPHM').\n
            default: 'IMRPhenomD'
        """
        return self._waveform_approximant

    @waveform_approximant.setter
    def waveform_approximant(self, value):
        self._waveform_approximant = value

    @property
    def frequency_domain_source_model(self):
        """
        Bilby frequency domain source model function.

        Returns
        -------
        frequency_domain_source_model : `str`
            Bilby frequency domain source model function.\n
            default: 'lal_binary_black_hole'
        """
        return self._frequency_domain_source_model

    @frequency_domain_source_model.setter
    def frequency_domain_source_model(self, value):
        self._frequency_domain_source_model = value

    @property
    def f_min(self):
        """
        Minimum waveform frequency.

        Returns
        -------
        f_min : `float`
            Minimum waveform frequency (Hz).\n
            default: 20.0
        """
        return self._f_min

    @f_min.setter
    def f_min(self, value):
        self._f_min = value

    @property
    def f_ref(self):
        """
        Reference frequency for waveform generation.

        Returns
        -------
        f_ref : `float`
            Reference frequency (Hz) for waveform generation.\n
            default: same as f_min
        """
        return self._f_ref

    @f_ref.setter
    def f_ref(self, value):
        self._f_ref = value

    @property
    def duration_max(self):
        """
        Maximum waveform duration.

        Returns
        -------
        duration_max : `float` or `None`
            Maximum waveform duration (s). Auto-set if None.\n
            default: None
        """
        return self._duration_max

    @duration_max.setter
    def duration_max(self, value):
        self._duration_max = value

    @property
    def duration_min(self):
        """
        Minimum waveform duration.

        Returns
        -------
        duration_min : `float` or `None`
            Minimum waveform duration (s). Auto-set if None.\n
            default: None
        """
        return self._duration_min

    @duration_min.setter
    def duration_min(self, value):
        self._duration_min = value

    @property
    def snr_method(self):
        """
        SNR calculation method.

        Returns
        -------
        snr_method : `str`
            SNR calculation method. Options: interpolation variants,
            inner_product variants, ann.\n
            default: 'interpolation_aligned_spins'
        """
        return self._snr_method

    @snr_method.setter
    def snr_method(self, value):
        self._snr_method = value

    @property
    def snr_type(self):
        """
        SNR type for calculations.

        Returns
        -------
        snr_type : `str`
            SNR type: 'optimal_snr' or 'observed_snr' (not implemented).\n
            default: 'optimal_snr'
        """
        return self._snr_type

    @snr_type.setter
    def snr_type(self, value):
        self._snr_type = value

    @property
    def noise_realization(self):
        """
        Noise realization for observed SNR.

        Returns
        -------
        noise_realization : `numpy.ndarray` or `None`
            Noise realization for observed SNR (not implemented).\n
            default: None
        """
        return self._noise_realization

    @noise_realization.setter
    def noise_realization(self, value):
        self._noise_realization = value

    @property
    def psds_list(self):
        """
        Detector power spectral densities. \n
        for the i-th detector: \n
            psds_list[i][0]: frequency (numpy.ndarray) \n
            psds_list[i][1]: power spectral density (numpy.ndarray) \n
            psds_list[i][2]: scipy.interpolate.interp1d object \n

        Returns
        -------
        psds_list : `list`
            List of PowerSpectralDensity objects for each detector.
        """
        return self._psds_list

    @psds_list.setter
    def psds_list(self, value):
        self._psds_list = value

    @property
    def detector_tensor_list(self):
        """
        Detector tensors for antenna response calculations.

        Returns
        -------
        detector_tensor_list : `list`
            List of numpy.ndarray detector tensors for antenna response.
        """
        return self._detector_tensor_list

    @detector_tensor_list.setter
    def detector_tensor_list(self, value):
        self._detector_tensor_list = value

    @property
    def detector_list(self):
        """
        Detector names.

        Returns
        -------
        detector_list : `list`
            List of detector names (e.g., ['H1', 'L1', 'V1']).
        """
        return self._detector_list

    @detector_list.setter
    def detector_list(self, value):
        self._detector_list = value

    @property
    def ifos(self):
        """
        Bilby interferometer objects.

        Returns
        -------
        ifos : `list`
            List of Bilby Interferometer objects.
        """
        return self._ifos

    @ifos.setter
    def ifos(self, value):
        self._ifos = value

    @property
    def interpolator_dir(self):
        """
        Directory for interpolation coefficient storage.

        Returns
        -------
        interpolator_dir : `str`
            Directory for interpolation coefficient storage.
            default: './interpolator_json'
        """
        return self._interpolator_dir

    @interpolator_dir.setter
    def interpolator_dir(self, value):
        self._interpolator_dir = value

    @property
    def path_interpolator(self):
        """
        Paths to interpolation coefficient files.

        Returns
        -------
        path_interpolator : `list`
            List of paths to interpolation coefficient files.
        """
        return self._path_interpolator

    @path_interpolator.setter
    def path_interpolator(self, value):
        self._path_interpolator = value

    @property
    def snr_partialsacaled_list(self):
        """
        Partial-scaled SNR interpolation coefficients.

        Returns
        -------
        snr_partialsacaled_list : `list`
            List of numpy.ndarray partial-scaled SNR interpolation coefficients.
        """
        return self._snr_partialsacaled_list

    @snr_partialsacaled_list.setter
    def snr_partialsacaled_list(self, value):
        self._snr_partialsacaled_list = value

    @property
    def multiprocessing_verbose(self):
        """
        Show progress bars for multiprocessing computations.

        Returns
        -------
        multiprocessing_verbose : `bool`
            Show progress bars for multiprocessing computations.\n
            default: True
        """
        return self._multiprocessing_verbose

    @multiprocessing_verbose.setter
    def multiprocessing_verbose(self, value):
        self._multiprocessing_verbose = value

    @property
    def identifier_dict(self):
        """
        Interpolator parameter dictionary for caching.

        Returns
        -------
        identifier_dict : `dict`
            Interpolator parameter dictionary for caching.
        """
        return self._identifier_dict

    @identifier_dict.setter
    def identifier_dict(self, value):
        self._identifier_dict = value

    @property
    def snr_th(self):
        """
        Individual detector SNR threshold.

        Returns
        -------
        snr_th : `float`
            Individual detector SNR threshold.\n
            default: 10.0
        """
        return self._snr_th

    @snr_th.setter
    def snr_th(self, value):
        self._snr_th = value

    @property
    def snr_th_net(self):
        """
        Network SNR threshold.

        Returns
        -------
        snr_th_net : `float`
            Network SNR threshold.\n
            default: 10.0
        """
        return self._snr_th_net

    @snr_th_net.setter
    def snr_th_net(self, value):
        self._snr_th_net = value

    @property
    def model_dict(self):
        """
        ANN models for each detector.

        Returns
        -------
        model_dict : `dict`
            ANN models for each detector (when snr_method='ann').
        """
        return self._model_dict

    @model_dict.setter
    def model_dict(self, value):
        self._model_dict = value

    @property
    def scaler_dict(self):
        """
        ANN feature scalers for each detector.

        Returns
        -------
        scaler_dict : `dict`
            ANN feature scalers for each detector (when snr_method='ann').
        """
        return self._scaler_dict

    @scaler_dict.setter
    def scaler_dict(self, value):
        self._scaler_dict = value

    @property
    def error_adjustment(self):
        """
        ANN error correction parameters.

        Returns
        -------
        error_adjustment : `dict`
            ANN error correction parameters (when snr_method='ann').
        """
        return self._error_adjustment

    @error_adjustment.setter
    def error_adjustment(self, value):
        self._error_adjustment = value

    @property
    def ann_catalogue(self):
        """
        ANN model configuration and paths.

        Returns
        -------
        ann_catalogue : `dict`
            ANN model configuration and paths (when snr_method='ann').
        """
        return self._ann_catalogue

    @ann_catalogue.setter
    def ann_catalogue(self, value):
        self._ann_catalogue = value

    @property
    def snr_recalculation(self):
        """
        Enable hybrid SNR recalculation near detection threshold.

        Returns
        -------
        snr_recalculation : `bool`
            Enable hybrid SNR recalculation near detection threshold.\n
            default: False
        """
        return self._snr_recalculation

    @snr_recalculation.setter
    def snr_recalculation(self, value):
        self._snr_recalculation = value

    @property
    def snr_recalculation_range(self):
        """
        SNR range triggering recalculation.

        Returns
        -------
        snr_recalculation_range : `list`
            SNR range [min, max] triggering recalculation.\n
            default: [6, 14]
        """
        return self._snr_recalculation_range

    @snr_recalculation_range.setter
    def snr_recalculation_range(self, value):
        self._snr_recalculation_range = value

    @property
    def snr_recalculation_waveform_approximant(self):
        """
        Waveform approximant for SNR recalculation.

        Returns
        -------
        snr_recalculation_waveform_approximant : `str`
            Waveform approximant for SNR recalculation.\n
            default: 'IMRPhenomXPHM'
        """
        return self._snr_recalculation_waveform_approximant

    @snr_recalculation_waveform_approximant.setter
    def snr_recalculation_waveform_approximant(self, value):
        self._snr_recalculation_waveform_approximant = value

    @property
    def get_interpolated_snr(self):
        """
        Interpolated SNR calculation function.

        Returns
        -------
        get_interpolated_snr : `function`
            Interpolated SNR calculation function (backend-specific).
        """
        return self._get_interpolated_snr

    @get_interpolated_snr.setter
    def get_interpolated_snr(self, value):
        self._get_interpolated_snr = value

    @property
    def noise_weighted_inner_product_jax(self):
        """
        JAX-accelerated inner product function.

        Returns
        -------
        noise_weighted_inner_product_jax : `function`
            JAX-accelerated inner product function (when snr_method='inner_product_jax').
        """
        return self._noise_weighted_inner_product_jax

    @noise_weighted_inner_product_jax.setter
    def noise_weighted_inner_product_jax(self, value):
        self._noise_weighted_inner_product_jax = value
