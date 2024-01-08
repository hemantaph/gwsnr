# -*- coding: utf-8 -*-
"""
This module contains functions for calculating the SNR of a CBC signal.
"""

import numpy as np
import bilby
from multiprocessing import Pool
from tqdm import tqdm

# from pycbc.detector import Detector
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

import utils
save_json = utils.save_json
NumpyEncoder = utils.NumpyEncoder
dealing_with_psds = utils.dealing_with_psds
interpolator_check = utils.interpolator_check
load_json = utils.load_json
save_json_dict = utils.save_json_dict
import njit_functions
get_interpolated_snr = njit_functions.get_interpolated_snr
findchirp_chirptime = njit_functions.findchirp_chirptime
antenna_response = njit_functions.antenna_response
antenna_response_array = njit_functions.antenna_response_array
import multiprocessing_routine
noise_weighted_inner_prod = multiprocessing_routine.noise_weighted_inner_prod

# defining constants
C = 299792458.0
G = 6.67408 * 1e-11
Mo = 1.989 * 1e30
Gamma = 0.5772156649015329
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
        options: 'interpolation', 'inner_product', 'pdet'
    psds : `dict`
        Dictionary of psds for different detectors. Default is None. If None, bilby's default psds will be used. Other options:\n
        Example 1: when values are psd name from pycbc analytical psds, psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044','V1':'AdvVirgo'}. To check available psd name run \n
        >>> import pycbc.psd
        >>> pycbc.psd.get_lalsim_psd_list()
        Example 2: when values are psd txt file available in bilby,
        psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt', 'K1':'KAGRA_design_asd.txt'}.
        For other psd files, check https://github.com/lscsoft/bilby/tree/master/bilby/gw/detector/noise_curves \n
        Example 3: when values are custom psd txt file. psds={'L1':'custom_psd.txt','H1':'custom_psd.txt'}. Custom created txt file has two columns. 1st column: frequency array, 2nd column: strain.
    isit_psd_file : `bool` or `dict`
        If set True, the given value of psds param should be of psds instead of asd. If asd, set isit_psd_file=False. Default is False. If dict, it should be of the form {'L1':True, 'H1':True, 'V1':True} and should have keys for all the detectors.
    psd_with_time : `bool` or `float`
        gps end time of strain data for which psd will be found. (this param will be given highest priority), example: psd_with_time=1246527224.169434. If False, psds given in psds param will be used. Default is False. If True (without gps time), psds will be calculated from strain data by setting gps end time as geocent_time-duration. Default is False.
    ifos : `list` or `None`
        List of interferometer objects. Default is None. If None, bilby's default interferometer objects will be used. For example for LIGO India detector, it can be defined as follows, \n
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
        >>> snr = GWSNR(psds=dict(LIO='your_asd_file.txt'), ifos=[ifosLIO], isit_psd_file=[False])
    interpolator_dir : `str`
        Path to store the interpolator pickle file. Default is './interpolator_pickle'.
    create_new_interpolator : `bool`
        If set True, new interpolator will be generated or replace the existing one. Default is False.
    gwsnr_verbose : `bool`
        If True, print all the parameters of the class instance. Default is True.

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

    snr_halfsacaled = None
    """``numpy.ndarray`` \n
    Array of half scaled SNR interpolation coefficients."""

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

    isit_psd_file = None
    """``dict`` \n
    dict keys with detector names and values as bool."""

    interpolator_dir = None
    """``str`` \n
    Path to store the interpolator pickle file."""

    detector_list = None
    """``list`` \n
    List of detectors."""

    stored_snrs = None
    """``dict`` \n
    Dictionary of stored SNRs."""

    def __init__(
        self,
        npool=int(4),
        mtot_min=2.0,
        mtot_max=184.0,
        ratio_min=0.1,
        ratio_max=1.0,
        mtot_resolution=100,
        ratio_resolution=100,
        sampling_frequency=2048.0,
        waveform_approximant="IMRPhenomD",
        minimum_frequency=20.0,
        snr_type="interpolation",
        psds=None,
        isit_psd_file=False,
        ifos=None,
        interpolator_dir="./interpolator_pickle",
        create_new_interpolator=False,
        gwsnr_verbose=True,
        multiprocessing_verbose=True,
    ):
        # setting instance attributes
        self.npool = npool

        # dealing with mtot_max
        mass_ratio = 1.
        func = lambda x: findchirp_chirptime(
            x / (1 + mass_ratio), x / (1 + mass_ratio) * mass_ratio, minimum_frequency
        )
        mtot_max_generated = fsolve(func, 150)[0]  # to make sure that chirptime is not negative, TaylorF2 might need this
        if mtot_max > mtot_max_generated:
            mtot_max = mtot_max_generated
        self.mtot_max = mtot_max

        self.mtot_min = mtot_min
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.mtot_resolution = mtot_resolution
        self.ratio_resolution = ratio_resolution
        # buffer of 0.01 is added to the ratio
        self.ratio_arr = np.geomspace(ratio_min-0.01, ratio_max+0.01, ratio_resolution)
        # buffer of 0.1 is added to the mtot
        self.mtot_arr = np.sort(
            mtot_min + mtot_max - np.geomspace(mtot_min-0.1, mtot_max+0.1, mtot_resolution)
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
        # self.isit_psd_file and self.detector_list are list of bool and list of strings respectively and will be set at the last.
        psds_list, detector_tensor_list, detector_list = dealing_with_psds(
            psds, isit_psd_file, ifos
        )

        # print some info
        self.print_all_params(gwsnr_verbose)

        # dealing with interpolator
        # interpolator check and generation will be skipped if snr_type="inner_product"
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
        
        # Note: it will only select detectors that does not have interpolator stored yet
        (
            self.psds_list,
            self.detector_tensor_list,
            self.detector_list,
            self.path_interpolator,
        ) = interpolator_check(
            param_dict_given=self.param_dict_given.copy(),
            interpolator_dir=interpolator_dir,
            create_new=create_new_interpolator,
        )

        # now generate interpolator, if not exists
        if snr_type != "inner_product":
            # len(detector_list) == 0, means all the detectors have interpolator stored
            if len(detector_list) > 0 and create_new_interpolator:
                print("Please be patient while the interpolator is generated")
                self.multiprocessing_verbose = False  # This lets multiprocessing to use map instead of imap_unordered function.
                self.init_halfscaled()
            else:
                # get all halfscaledSNR from the stored interpolator
                self.snr_halfsacaled_list = []
                for j in range(len(detector_list)):
                    self.snr_halfsacaled_list.append(
                        load_json(self.path_interpolator[j])
                    )

        # now the entire detector_list
        self.psds_list = psds_list
        self.detector_tensor_list = detector_tensor_list
        self.detector_list = detector_list
        self.multiprocessing_verbose = multiprocessing_verbose

    def print_all_params(self, verbose=True):
        """
        Function to print all the parameters of the class instance

        Parameters
        ----------
        verbose : `bool`
            If True, print all the parameters of the class instance. Default is True.
        """

        if verbose:
            print("npool: ", self.npool)
            print("snr_type: ", self.snr_type)
            print("waveform_approximant: ", self.waveform_approximant)
            print("sampling_frequency: ", self.sampling_frequency)
            if self.snr_type == "interpolation":
                print("mtot_min: ", self.mtot_min)
                print("mtot_max: ", self.mtot_max)
                print("ratio_min: ", self.ratio_min)
                print("ratio_max: ", self.ratio_max)
                print("mtot_resolution: ", self.mtot_resolution)
                print("ratio_resolution: ", self.ratio_resolution)
                print("interpolator_dir: ", self.interpolator_dir)

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
        jsonFile=False,
    ):
        """
        -----------------
        Input parameters (GW parameters)
        -----------------
        mass_1               : Heavier compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        mass_2               : Lighter compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        luminosity_distance  : Distance between detector and binary, unit: Mpc
        theta_jn                 : Inclination angle of binary orbital plane wrt to the line of sight. unit: rad
        psi                  : Polarization angle. unit: rad
        phase                : Phase of GW at the the time of coalesence, unit: rad
        geocent_time         : GPS time of colescence of that GW, unit: sec
        ra                   : Right ascention of source position, unit: rad
        dec                  : Declination of source position, unit: rad
        -----------------
        Return values
        -----------------
        snr_dict              : dictionary containing net optimal snr and optimal snr of individual detectors
                               example of optimal_snr_unscaled return values for len(mass_1)=3
                                {'optimal_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
                                 'L1': array([132.08275995, 205.04492349, 246.47822334]),
                                 'H1': array([ 84.00372897, 130.40716432, 156.75845871])}

        """
        if gw_param_dict != False:
            mass_1 = gw_param_dict["mass_1"]
            mass_2 = gw_param_dict["mass_2"]
            luminosity_distance = gw_param_dict["luminosity_distance"]
            theta_jn = gw_param_dict["theta_jn"]
            psi = gw_param_dict["psi"]
            phase = gw_param_dict["phase"]
            geocent_time = gw_param_dict["geocent_time"]
            ra = gw_param_dict["ra"]
            dec = gw_param_dict["dec"]
            size = len(mass_1)
            try:
                a_1 = gw_param_dict["a_1"]
                a_2 = gw_param_dict["a_2"]
                try:
                    tilt_1 = gw_param_dict["tilt_1"]
                    tilt_2 = gw_param_dict["tilt_2"]
                    phi_12 = gw_param_dict["phi_12"]
                    phi_jl = gw_param_dict["phi_jl"]
                except:
                    tilt_1 = np.zeros(size)
                    tilt_2 = np.zeros(size)
                    phi_12 = np.zeros(size)
                    phi_jl = np.zeros(size)
            except:
                a_1 = np.zeros(size)
                a_2 = np.zeros(size)
                tilt_1 = np.zeros(size)
                tilt_2 = np.zeros(size)
                phi_12 = np.zeros(size)
                phi_jl = np.zeros(size)

        # save geocent_time in json file
        # with open("./geocent_time.json", "w") as write_file:
        #     json.dump(list(geocent_time), write_file, indent=4)

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
            )
        else:
            if self.snr_type == "inner_product":
                print("solving SNR with inner product")
            else:
                print(
                    "SNR function type not recognised, using inner_product method instead"
                )
            snr_dict = self.compute_bilby_snr_(
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
            )
        return snr_dict

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
        -----------------
        Input parameters (GW parameters)
        -----------------
        mass_1               : Heavier compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        mass_2               : Lighter compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        luminosity_distance  : Distance between detector and binary, unit: Mpc
        theta_jn                 : Inclination angle of binary orbital plane wrt to the line of sight. unit: rad
        psi                  : Polarization angle. unit: rad
        phase                : Phase of GW at the the time of coalesence, unit: rad
        geocent_time         : GPS time of colescence of that GW, unit: sec
        ra                   : Right ascention of source position, unit: rad
        dec                  : Declination of source position, unit: rad
        -----------------
        Return values
        -----------------
        optimal_snr              : dictionary containing net optimal snr and optimal snr of individual detectors
                               example of optimal_snr_unscaled return values for len(mass_1)=3
                                {'optimal_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
                                 'L1': array([132.08275995, 205.04492349, 246.47822334]),
                                 'H1': array([ 84.00372897, 130.40716432, 156.75845871])}

        """
        
        detector_tensor = np.array(self.detector_tensor_list)
        detectors = np.array(self.detector_list)
        snr_halfscaled = np.array(self.snr_halfsacaled_list)
        size = len(mass_1)
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec = (
            np.array([luminosity_distance]).reshape(-1) * np.ones(size),
            np.array([theta_jn]).reshape(-1) * np.ones(size),
            np.array([psi]).reshape(-1) * np.ones(size),
            np.array([phase]).reshape(-1) * np.ones(size),
            np.array([geocent_time]).reshape(-1) * np.ones(size),
            np.array([ra]).reshape(-1) * np.ones(size),
            np.array([dec]).reshape(-1) * np.ones(size),
        )

        mtot = mass_1 + mass_2
        idx2 = (mtot >= self.mtot_min) & (mtot <= self.mtot_max)
        idx_tracker = np.arange(size)
        idx_tracker = idx_tracker[idx2]  # idx_tracker is the index of the required masses
        size_ = len(idx_tracker)
        if size_ == 0:
            print(
                "mass_1 and mass_2 must be within the range of mtot_min and mtot_max"
            )
            raise ValueError
        # all other parameters should be of the same size
        # this lessen the number of times the loop will run
        mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec = mass_1[idx2], mass_2[idx2], luminosity_distance[idx2], theta_jn[idx2], psi[idx2], phase[idx2], geocent_time[idx2], ra[idx2], dec[idx2]

        # get interpolated snr
        snr, snr_effective = get_interpolated_snr(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, detector_tensor, snr_halfscaled, self.ratio_arr, self.mtot_arr)

        optimal_snr = dict()
        for j, det in enumerate(detectors):
            optimal_snr[det] = snr[j]
        optimal_snr["optimal_snr_net"] = snr_effective

        # saving as json file
        if output_jsonfile:
            if isinstance(output_jsonfile, str):
                save_json_dict(optimal_snr, output_jsonfile)
            else:
                save_json_dict(optimal_snr, "snr.json")

        return optimal_snr

    def bns_horizon(self, snr_threshold=8.0):
        """
        Function for finding detector horizon distance for BNS (m1=m2=1.4)

        Parameters
        ----------
        snr_threshold : float
            SNR threshold for the horizon distance
            default: 8.

        Returns
        -------
        horizon : array
        """

        detectors = self.detector_list.copy()

        # geocent_time cannot be array here
        # this geocent_time is only to get halfScaledSNR
        geocent_time_ = 1246527224.169434  # random time from O3

        theta_jn_, ra_, dec_, psi_, phase_ = 0.0, 0.0, 0.0, 0.0, 0.0
        luminosity_distance_ = 100.0
        q = 1.0
        mass_1_ = 1.4
        mass_2_ = mass_1_ * q

        ######## calling bilby_snr ########
        optimal_snr_unscaled = self.compute_bilby_snr_(
            mass_1=mass_1_,
            mass_2=mass_2_,
            luminosity_distance=luminosity_distance_,
            theta_jn=theta_jn_,
            psi=psi_,
            phase=phase_,
            ra=ra_,
            dec=dec_,
        )

        ######## filling in interpolation table for different detectors ########
        horizon = np.zeros(len(detectors))
        for j in range(len(detectors)):
            Fp = self.ifos[j].antenna_response(ra_, dec_, geocent_time_, psi_, "plus")
            Fc = self.ifos[j].antenna_response(ra_, dec_, geocent_time_, psi_, "cross")
            Deff2 = luminosity_distance_ / np.sqrt(
                Fp**2 * ((1 + np.cos(theta_jn_) ** 2) / 2) ** 2
                + Fc**2 * np.cos(theta_jn_) ** 2
            )

            horizon[j] = (Deff2 / snr_threshold) * optimal_snr_unscaled[detectors[j]]

        return horizon

    def init_halfscaled(self):
        """

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        mtot_min = self.mtot_min
        detectors = self.detector_list.copy()
        detector_tensor = self.detector_tensor_list.copy()
        num_det = np.arange(len(detectors), dtype=int)

        # This distribution is used to get more points at higher mass
        mtot_table = self.mtot_arr
        print(f"Generating interpolator for {detectors} detectors")

        try:
            if mtot_min < 1.0:
                raise ValueError
        except ValueError:
            print("Error: mass too low")

        # geocent_time cannot be array here
        # this geocent_time is only to get halfScaledSNR
        geocent_time_ = 1246527224.169434  # random time from O3
        theta_jn_, ra_, dec_, psi_, phase_ = 0.0, 0.0, 0.0, 0.0, 0.0
        luminosity_distance_ = 100.0

        # Effective luminosity distance
        dl_eff = np.zeros(len(num_det))
        for j in num_det:
            Fp = antenna_response(ra_, dec_, geocent_time_, psi_, detector_tensor[j], "plus")  # plus polarization
            Fc = antenna_response(ra_, dec_, geocent_time_, psi_, detector_tensor[j], "cross")  # cross polarization
            dl_eff[j] = luminosity_distance_ / np.sqrt(
                Fp**2 * ((1 + np.cos(theta_jn_) ** 2) / 2) ** 2
                + Fc**2 * np.cos(theta_jn_) ** 2
            )

        ratio = self.ratio_arr.copy()
        snr_half_ = []
        i = 0
        # interpolation along mtot for each mass_ratio
        for q in tqdm(
            ratio,
            desc="interpolation for each mass_ratios",
            total=len(ratio),
            ncols=100,
        ):
            mass_1_ = mtot_table / (1 + q)
            mass_2_ = mass_1_ * q
            mchirp = ((mass_1_ * mass_2_) ** (3 / 5)) / ((mtot_table) ** (1 / 5))
            ######## calling bilby_snr ########
            optimal_snr_unscaled = self.compute_bilby_snr_(
                mass_1=mass_1_,
                mass_2=mass_2_,
                luminosity_distance=luminosity_distance_,
                theta_jn=theta_jn_,
                psi=psi_,
                phase=phase_,
                ra=ra_,
                dec=dec_,
            )

            a2 = mchirp ** (5.0 / 6.0)
            # filling in interpolation table for different detectors
            snr_half_buffer = []
            for j in num_det:
                snr_half_buffer.append(CubicSpline(
                    mtot_table, (dl_eff[j] / a2) * optimal_snr_unscaled[detectors[j]]
                ).c)
            snr_half_.append(np.array(snr_half_buffer))
            i += 1  # iterator over mass_ratio
        snr_half_ = np.array(snr_half_)

        # save the interpolators for each detectors
        snr_half_buffer = []
        for j in num_det:
            save_json(snr_half_[:,j], self.path_interpolator[j])
            snr_half_buffer.append(snr_half_[:,j])

        self.snr_halfsacaled_list = snr_half_buffer

    def compute_bilby_snr_(
        self,
        mass_1,
        mass_2,
        luminosity_distance=100.0,
        theta_jn=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=np.array([]),
        ra=0.0,
        dec=0.0,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
    ):
        """
        SNR calculated using bilby python package
        Use for interpolation purpose

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
        -------
        snr_dict : dict
            dictionary containing net optimal snr and optimal snr of individual detectors
            example of optimal_snr_unscaled return values for len(mass_1)=3
            {'optimal_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
            'L1': array([132.08275995, 205.04492349, 246.47822334]),
            'H1': array([ 84.00372897, 130.40716432, 156.75845871])}
        """

        npool = self.npool
        geocent_time_ = 1246527224.169434  # random time from O3
        # check whether there is input for geocent_time
        if not np.array(geocent_time).tolist():
            geocent_time = geocent_time_
        sampling_frequency = self.sampling_frequency
        detectors = self.detector_list.copy()
        detector_tensor = np.array(self.detector_tensor_list.copy())
        approximant = self.waveform_approximant
        f_min = self.f_min
        num_det = np.arange(len(detectors), dtype=int)

        # get the psds for the required detectors
        psd_dict = {detectors[i]: self.psds_list[i] for i in num_det}

        #############################################
        # setting up parameters for multiprocessing #
        #############################################
        # reshape(-1) is so that either a float value is given or the input is an numpy array
        # np.ones is multipled to make sure everything is of same length
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        num = len(mass_1)
        # reshaping other parameters
        (
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
        ) = (
            np.array([luminosity_distance]).reshape(-1) * np.ones(num),
            np.array([theta_jn]).reshape(-1) * np.ones(num),
            np.array([psi]).reshape(-1) * np.ones(num),
            np.array([phase]).reshape(-1) * np.ones(num),
            np.array([geocent_time]).reshape(-1) * np.ones(num),
            np.array([ra]).reshape(-1) * np.ones(num),
            np.array([dec]).reshape(-1) * np.ones(num),
            np.array([a_1]).reshape(-1) * np.ones(num),
            np.array([a_2]).reshape(-1) * np.ones(num),
            np.array([tilt_1]).reshape(-1) * np.ones(num),
            np.array([tilt_2]).reshape(-1) * np.ones(num),
            np.array([phi_12]).reshape(-1) * np.ones(num),
            np.array([phi_jl]).reshape(-1) * np.ones(num),
        )

        # IMPORTANT: time duration calculation for each of the mass combination
        safety = 1.2
        approx_duration = safety * findchirp_chirptime(mass_1, mass_2, f_min)
        duration = np.ceil(approx_duration + 2.0)

        #############################################
        # setting up parameters for multiprocessing #
        #############################################
        size1 = len(mass_1)
        iterations = np.arange(size1)  # to keep track of index

        dectector_arr = np.array(detectors) * np.ones(
            (size1, len(detectors)), dtype=object
        )
        psds_dict_list = np.array([np.full(size1, psd_dict, dtype=object)]).T

        input_arguments = np.array(
            [
                mass_1,
                mass_2,
                luminosity_distance,
                theta_jn,
                psi,
                phase,
                ra,
                dec,
                geocent_time,
                a_1,
                a_2,
                tilt_1,
                tilt_2,
                phi_12,
                phi_jl,
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
        hp_inner_hp = np.zeros((len(num_det),size1), dtype=np.complex128)
        hc_inner_hc = np.zeros((len(num_det),size1), dtype=np.complex128)
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
                    hp_inner_hp[:,iter_i] = hp_inner_hp_i
                    hc_inner_hc[:,iter_i] = hc_inner_hc_i
            else:
                # with map, without tqdm
                for result in pool.map(
                    noise_weighted_inner_prod, input_arguments
                ):
                    hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                    hp_inner_hp[:,iter_i] = hp_inner_hp_i
                    hc_inner_hc[:,iter_i] = hc_inner_hc_i

        # get polarization tensor
        # np.shape(Fp) = (size1, len(num_det))
        Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

        snr_dict = dict()
        snrs_sq = abs((Fp**2) * hp_inner_hp + (Fc**2) * hc_inner_hc)
        # for individual detectors
        snr_dict = {detectors[j]: np.sqrt(snrs_sq[j]) for j in num_det}
        # net snr
        snr_dict["optimal_snr_net"] = np.sqrt(np.sum(snrs_sq, axis=0))
        self.stored_snrs = snr_dict  # this stored snrs can be use for Pdet calculation

        return snr_dict

    ####################################################
    #                                                  #
    #            psd array finder from pycbc           #
    #                                                  #
    ####################################################
    def power_spectral_density_pycbc(self, psd):
        """
        psd array finder from pycbc

        Parameters
        ----------
        psd : str
            name of the psd
            e.g. 'aLIGOaLIGODesignSensitivityT1800044'

        Returns
        -------
        psd_array : bilby.gw.detector.psd.PowerSpectralDensity object
        """
        import pycbc
        import pycbc.psd

        delta_f = 1.0 / 16.0
        flen = int(self.sampling_frequency / delta_f)
        low_frequency_cutoff = self.f_min
        psd_ = pycbc.psd.from_string(psd, flen, delta_f, low_frequency_cutoff)
        return bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd_.sample_frequencies, psd_array=psd_.data
        )

    ####################################################
    #                                                  #
    #             Probaility of detection              #
    #                                                  #
    ####################################################
    def pdet(self, snr_dict, rho_th=8.0, rhoNet_th=8.0):
        """
        Probaility of detection of GW for the given sensitivity of the detectors
        -----------------
        Input parameters
        -----------------
        snrs      : Signal-to-noise ratio for all the chosen detectors and GW parameters
                    (numpy array of float)

        -----------------
        Return values
        -----------------
        dict_pdet  : dictionary of {'pdet_net':pdet_net, 'pdet_L1':pdet_L1, 'pdet_H1':pdet_H1, 'pdet_V1':pdet_V1}
        """

        detectors = np.array(self.detector_list)
        pdet_dict = {}
        for det in detectors:
            pdet_dict["pdet_" + det] = 1 - norm.cdf(rho_th - snr_dict[det])

        pdet_dict["pdet_net"] = 1 - norm.cdf(rhoNet_th - snr_dict["optimal_snr_net"])

        return pdet_dict
