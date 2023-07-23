# for delta_f =1/duration, duration = 16s
# f_min =20Hz
# duration=16.0, sampling_frequency=4096,
# note: setting mtot_min and mtot_max is important.
# mtot_min=219. is accordance to minimum_frequency = 20
# __init__ paramters are important don't to change for a particular analysis
# they are detector and waveform dependent parameters
# at f_min==10Hz: mtot_max=439.6
import numpy as np
import bilby
import pycbc
import pycbc.psd
# from pycbc.detector import Detector
from scipy.stats import norm
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
from scipy.optimize import fsolve
from multiprocessing import Pool
from tqdm import tqdm
import json
import os
import pickle

C = 299792458.0
G = 6.67408 * 1e-11
Mo = 1.989 * 1e30
Gamma = 0.5772156649015329
Pi = np.pi
MTSUN_SI = 4.925491025543576e-06

"""
------------------------------------------------
    class containing following methods
    1. to calculate fast SNR
    2. interpolation of with cubic spline
    with bilby SNR
    3. Pdet: probability of detection
------------------------------------------------
"""


class GWSNR:
    ####################################################
    #                                                  #
    #             Class initialization                 #
    #                                                  #
    ####################################################
    def __init__(
        self,
        npool=int(4),
        mtot_min=2.0,
        mtot_max=439.6,
        nsamples_mtot=100,
        nsamples_mass_ratio=50,
        sampling_frequency=2048.0,
        waveform_approximant="IMRPhenomD",
        minimum_frequency=20.0,
        snr_type="interpolation",
        waveform_inspiral_must_be_above_fmin=False,
        psds=None,
        psd_file=False,
        ifos=None,
        interpolator_dir="./interpolator_pickle",
    ):
        """
        Initialized parameters and functions
        snr_half_scaled() : function for finding (f/PSD) integration in the limit [f_min,f_max]
        list_of_detectors :  list of detector initials, e.g. L1 for Livingston
        f_min : minimum frequency for the detector
        -----------------
        input parameters
        -----------------
        mtot_min           : minimum value of Mtotal=mass_1+mass_2, use in interpolation
        mtot_max           : maximum value of Mtotal=mass_1+mass_2, use in interpolation
        nsamples           : number of points you want to use for SNR interpolation (here it is half SNR not complete)
        list_of_detectors  : detector list. It can be single or multiple.
        duration           : duration of the data in time domain. 
        sampling_frequency : sampling frequency of the data. e.g. 4096Hz,2048Hz,1024Hz
        waveform_arguments : contains which waveform model to use for interpolation. Extra paramters like reference_frequency\
                             minimum_frequency are also included. minimum_frequency will also relate to the mtot_max set inside\
                             the code. High mass blackholes tends to merge at lower frequency < f_min, and can have SNR=0
        snr_type           : method for SNR calculation. Values: 'interpolation', 'inner_product' 
        
        psds               : psd dict.
                               example_1=> when values are psd name from pycbc analytical psds, 
                               psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044','V1':'AdvVirgo'}
                               to check available psd name run $ import pycbc.psd ; $ pycbc.psd.get_lalsim_psd_list()
                               example_2=> when values are psd txt file in bilby,
                               psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt', 'K1':'KAGRA_design_asd.txt'}
                               For other psd files, check https://github.com/lscsoft/bilby/tree/master/bilby/gw/detector/noise_curves
                               example_3=> when values are custom psd txt file.
                               psds={'L1':'custom_psd.txt','H1':'custom_psd.txt'}
                               custom created txt file has two columns. 1st column: frequency array, 2nd column: strain 
        psd_file           : if set True, the given value of psds param should be of psds instead of asd. If asd, set psd_file=False.
        
        """
        self.npool = npool
        self.mtot_min = mtot_min
        self.mtot_max = mtot_max
        self.nsamples = nsamples_mtot
        ratio = np.geomspace(0.1, 1, nsamples_mass_ratio)
        self.ratio = ratio

        self.sampling_frequency = sampling_frequency
        self.waveform_approximant = waveform_approximant
        self.f_min = minimum_frequency
        self.waveform_type = self.waveform_classifier(waveform_approximant)
        self.snr_type = snr_type
        self.waveform_inspiral_must_be_above_fmin = waveform_inspiral_must_be_above_fmin

        if not psds:
            print("psds not given. Choosing bilby's default psds")
            psds = dict()
            psds["L1"] = "aLIGO_O4_high_asd.txt"
            psds["H1"] = "aLIGO_O4_high_asd.txt"
            psds["V1"] = "AdV_asd.txt"
            self.psds = psds
            list_of_detectors = list(psds.keys())
            psd_file = [False, False, False]
            # for Fp, Fc calculation
            self.ifos = bilby.gw.detector.InterferometerList(list_of_detectors)
        else:
            self.psds = psds
            list_of_detectors = list(psds.keys())
            psd_file = (
                np.array([psd_file]).reshape(-1) * np.ones(len(list_of_detectors))
            ).astype("bool")
            # for Fp, Fc calculation
            ifos_ = []
            len_ = len(list_of_detectors)
            for i in range(len_):
                try:
                    if ifos[i]:
                        ifos_.append(ifos[i])
                    else:
                        ifos_.append(
                            bilby.gw.detector.InterferometerList(
                                [list_of_detectors[i]]
                            )[0]
                        )
                except:
                    ifos_.append(
                        bilby.gw.detector.InterferometerList([list_of_detectors[i]])[0]
                    )

            self.ifos = ifos_
        print("given psds: ", psds)
        self.psd_file = psd_file

        # dealing with interpolator
        if snr_type == "interpolation":
            self.interpolator_dict = {}
            self.list_of_detectors = []
            psd_file_buffer = []

            # getting interpolator if exists
            k = 0  # keep track of psd_file_buffer param
            for det in list_of_detectors:
                path_interpolator, it_exist = self.interpolator_pickle_path(
                    det, list_of_detectors, interpolator_dir
                )
                if it_exist:
                    print(
                        f"Interpolator will be loaded for {det} detector from {path_interpolator}"
                    )
                else:
                    print(
                        f"Interpolator will be generated for {det} detector at {path_interpolator}"
                    )
                    self.list_of_detectors.append(det)
                    psd_file_buffer.append(psd_file[k])

                self.interpolator_dict[det] = path_interpolator

                k += 1  # keep track of psd_file_buffer param

            # generating new interpolator
            if len(self.list_of_detectors) > 0:
                self.psd_file = psd_file_buffer
                self.init_halfScaled()
                self.psd_file = psd_file
                print("interpolator generated")

        # now the entire list_of_detectors
        self.list_of_detectors = list_of_detectors

        return None

    ####################################################
    #                                                  #
    #      store or get interpolator pickle path       #
    #                                                  #
    ####################################################
    def interpolator_pickle_path(
        self, detector, detector_list, path="./interpolator_pickle"
    ):
        """
        Function for storing or getting interpolator pickle path

        Parameters
        ----------
        detector : str
            detector name
            e.g. 'L1'
        detector_list : list
            list of detectors
            e.g. ['L1','H1','V1']
        path : str
            path to store the pickle file
            default: './interpolator_pickle'

        Returns
        -------
        path_interpolator : str
            path to the interpolator pickle file
            e.g. './interpolator_pickle/L1/halfSNR_dict_0.pickle'
        it_exist: bool
            True if the interpolator exists
            False if the interpolator does not exists
        """

        # arg no. for the detector from the detector list
        det_arg = detector_list.index(detector)
        det_path = path + "/" + detector

        # check if path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # check if detector path exists
        if not os.path.exists(det_path):
            os.makedirs(det_path)

        # check if param_dict_list.pickle exists
        if not os.path.exists(det_path + "/param_dict_list.pickle"):
            dict_list = []
            with open(det_path + "/param_dict_list.pickle", "wb") as handle:
                pickle.dump(dict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # for checking existing interpolator pickle in the det_path/param_dict_list.pickle file
        param_dict_stored = pickle.load(
            open(det_path + "/param_dict_list.pickle", "rb")
        )
        param_dict_given = {
            "mtot_min": self.mtot_min,
            "mtot_max": self.mtot_max,
            "nsamples_mtot": self.nsamples,
            "sampling_frequency": self.sampling_frequency,
            "waveform_approximant": self.waveform_approximant,
            "minimum_frequency": self.f_min,
            "waveform_inspiral_must_be_above_fmin": self.waveform_inspiral_must_be_above_fmin,
            "psds": self.psds[detector],
            "psd_file": self.psd_file[det_arg],
            "ifos": str(self.ifos[det_arg]),
        }

        len_ = len(param_dict_stored)
        if param_dict_given in param_dict_stored:
            # try and except is added so that user can regenerate a new interpolator pickle file just by
            # deleting the right file and reruing gwsnr with that params again
            # also, if the user delete the file by mistake, it will generate in the next run
            idx = param_dict_stored.index(param_dict_given)
            # check if interpolator pickle exists
            # get halfSNR interpolator if exists
            path_interpolator = det_path + "/halfSNR_dict_" + str(idx) + ".pickle"
            # there will be exception if the file is deleted by mistake
            if os.path.exists(path_interpolator):
                it_exist = True
            else:
                it_exist = False

        # if related dict not found in the param_dict_list.pickle
        else:
            it_exist = False
            path_interpolator = det_path + "/halfSNR_dict_" + str(len_) + ".pickle"
            # print("related dict not found in the param_dict_list.pickle, new interpolator will be generated")

            # store the pickle dict
            param_dict_stored.append(param_dict_given)
            with open(det_path + "/param_dict_list.pickle", "wb") as handle:
                pickle.dump(param_dict_stored, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print(f"In case if you need regeneration of interpolator of the given gwsnr param, please delete this file, {path_interpolator} \n")

        return (path_interpolator, it_exist)

    ####################################################
    #                                                  #
    #             waveform classifier                  #
    #                                                  #
    ####################################################
    def waveform_classifier(self, waveform_approximant):
        waveform_dict = {
            "Inspiral": ["TaylorF2", "TaylorF2Ecc"],
            "IMR": ["IMRPhenomD", "IMRPhenomXPHM"],
            "Ringdown": [],
        }
        if waveform_approximant in waveform_dict["Inspiral"]:
            print(f"Given: Inspiral waveform, {self.waveform_approximant}.")
            return "Inspiral"
        elif waveform_approximant in waveform_dict["IMR"]:
            print("Given: IMR waveform, {self.waveform_approximant}.")
            return "IMR"
        else:
            print(
                "waveform type not recognised. It will be considered as IMRPhenom waveform"
            )

    ####################################################
    #                                                  #
    #             Main SNR finder function             #
    #                                                  #
    ####################################################
    def snr(
        self,
        mass_1=10.0,
        mass_2=10.0,
        luminosity_distance=100.0,
        iota=0.0,
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
        GWparam_dict=False,
        verbose=True,
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
        iota                 : Inclination angle of binary orbital plane wrt to the line of sight. unit: rad
        psi                  : Polarization angle. unit: rad
        phase                : Phase of GW at the the time of coalesence, unit: rad
        geocent_time         : GPS time of colescence of that GW, unit: sec
        ra                   : Right ascention of source position, unit: rad
        dec                  : Declination of source position, unit: rad
        -----------------
        Return values
        -----------------
        snr_dict              : dictionary containing net optimal snr and optimal snr of individual detectors
                               example of opt_snr_unscaled return values for len(mass_1)=3
                                {'opt_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
                                 'L1': array([132.08275995, 205.04492349, 246.47822334]),
                                 'H1': array([ 84.00372897, 130.40716432, 156.75845871])}

        """
        if GWparam_dict != False:
            mass_1 = GWparam_dict["mass_1"]
            mass_2 = GWparam_dict["mass_2"]
            luminosity_distance = GWparam_dict["luminosity_distance"]
            iota = GWparam_dict["iota"]
            psi = GWparam_dict["psi"]
            phase = GWparam_dict["phase"]
            geocent_time = GWparam_dict["geocent_time"]
            ra = GWparam_dict["ra"]
            dec = GWparam_dict["dec"]
            a_1 = GWparam_dict["a_1"]
            a_2 = GWparam_dict["a_2"]
            tilt_1 = GWparam_dict["tilt_1"]
            tilt_2 = GWparam_dict["tilt_2"]
            phi_12 = GWparam_dict["phi_12"]
            phi_jl = GWparam_dict["phi_jl"]

        if self.snr_type == "interpolation":
            snr_dict = self.snr_with_interpolation(
                mass_1,
                mass_2,
                luminosity_distance=luminosity_distance,
                iota=iota,
                psi=psi,
                phase=phase,
                geocent_time=geocent_time,
                ra=ra,
                dec=dec,
                jsonFile=jsonFile,
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
                theta_jn=iota,
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
                jsonFile=jsonFile,
            )
        return snr_dict

    ####################################################
    #                                                  #
    #   fast snr with cubic spline interpolation       #
    #                                                  #
    ####################################################
    def snr_with_interpolation(
        self,
        mass_1,
        mass_2,
        luminosity_distance=100.0,
        iota=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=1246527224.169434,
        ra=0.0,
        dec=0.0,
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
        iota                 : Inclination angle of binary orbital plane wrt to the line of sight. unit: rad
        psi                  : Polarization angle. unit: rad
        phase                : Phase of GW at the the time of coalesence, unit: rad
        geocent_time         : GPS time of colescence of that GW, unit: sec
        ra                   : Right ascention of source position, unit: rad
        dec                  : Declination of source position, unit: rad
        -----------------
        Return values
        -----------------
        opt_snr              : dictionary containing net optimal snr and optimal snr of individual detectors
                               example of opt_snr_unscaled return values for len(mass_1)=3
                                {'opt_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
                                 'L1': array([132.08275995, 205.04492349, 246.47822334]),
                                 'H1': array([ 84.00372897, 130.40716432, 156.75845871])}

        """
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        size = len(mass_1)
        luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec = (
            np.array([luminosity_distance]).reshape(-1) * np.ones(size),
            np.array([iota]).reshape(-1) * np.ones(size),
            np.array([psi]).reshape(-1) * np.ones(size),
            np.array([phase]).reshape(-1) * np.ones(size),
            np.array([geocent_time]).reshape(-1) * np.ones(size),
            np.array([ra]).reshape(-1) * np.ones(size),
            np.array([dec]).reshape(-1) * np.ones(size),
        )

        Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
        mtot = mass_1 + mass_2
        luminosity_distance = luminosity_distance

        snr_half_scaled = np.zeros(size)
        # select only those that have inspiral part above f_min
        if self.waveform_inspiral_must_be_above_fmin == True:
            approx_duration = self.findchirp_chirptime(mass_1, mass_2, self.f_min)
            idx2 = approx_duration > 0.0
        else:
            idx2 = np.full(size, True)

        idx2 = idx2 & (mtot >= self.mtot_min) & (mtot <= self.mtot_max)

        A1 = Mc ** (5.0 / 6.0)
        ci_2 = np.cos(theta_jn) ** 2
        ci_param = ((1 + np.cos(theta_jn) ** 2) / 2) ** 2
        detectors = self.list_of_detectors.copy()

        opt_snr = {"opt_snr_net": 0}

        idx_ratio = np.searchsorted(self.ratio, mass_2 / mass_1)
        idx_tracker = np.arange(size)
        idx_tracker = idx_tracker[idx2]
        # self.idx_ratio = idx_ratio
        # loop wrt detectors
        for i in range(len(detectors)):
            det = detectors[i]
            # get interpolated half_snr for the required detector
            with open(self.interpolator_dict[det], "rb") as handle:
                halfSNR_interpolator = pickle.load(handle)

            # calculation of snr_half_scaled for particular detector at the required mtot
            for j in idx_tracker:
                snr_half_scaled[j] = halfSNR_interpolator[idx_ratio[j]](
                    mtot[j]
                )  # i is iterator wrt detectors

            Deff1 = np.zeros(size)
            for k in range(len(ra)):
                Fp = self.ifos[i].antenna_response(
                    ra[k], dec[k], geocent_time[k], psi[k], "plus"
                )
                Fc = self.ifos[i].antenna_response(
                    ra[k], dec[k], geocent_time[k], psi[k], "cross"
                )
                Deff1[k] = luminosity_distance[k] / np.sqrt(
                    Fp**2 * ci_param[k] + Fc**2 * ci_2[k]
                )

            opt_snr[det] = (A1 / Deff1) * snr_half_scaled
            opt_snr["opt_snr_net"] += opt_snr[det] ** 2

        opt_snr["opt_snr_net"] = np.sqrt(opt_snr["opt_snr_net"])
        self.stored_snrs = opt_snr  # this stored snrs can be use for Pdet calculation

        # saving as json file
        if jsonFile:
            parameters_dict = {
                "mass_1": mass_1,
                "mass_2": mass_2,
                "luminosity_distance": luminosity_distance,
                "theta_jn": theta_jn,
                "psi": psi,
                "phase": phase,
                "ra": ra,
                "dec": dec,
                "geocent_time": geocent_time,
            }
            parameters_dict.update(opt_snr)
            file_name = "./bilby_GWparams_interpolatedSNRs.json"
            json_dump = json.dumps(parameters_dict, cls=NumpyEncoder)
            with open(file_name, "w") as write_file:
                json.dump(json.loads(json_dump), write_file, indent=4)

        # how to load data form .json file
        # f = open ('data.json', "r")
        # data = json.loads(f.read())

        return opt_snr

    ####################################################
    #                                                  #
    #   half_snr vs mtot table for interpolation       #
    #                                                  #
    ####################################################
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

        detectors = self.list_of_detectors.copy()

        # geocent_time cannot be array here
        # this geocent_time is only to get halfScaledSNR
        geocent_time_ = 1246527224.169434  # random time from O3

        iota_, ra_, dec_, psi_, phase_ = 0.0, 0.0, 0.0, 0.0, 0.0
        luminosity_distance_ = 100.0
        q = 1.0
        mass_1_ = 1.4
        mass_2_ = mass_1_ * q

        ######## calling bilby_snr ########
        opt_snr_unscaled = self.compute_bilby_snr_(
            mass_1=mass_1_,
            mass_2=mass_2_,
            luminosity_distance=luminosity_distance_,
            theta_jn=iota_,
            psi=psi_,
            ra=ra_,
            dec=dec_,
            verbose=False,
            jsonFile=False,
        )

        ######## filling in interpolation table for different detectors ########
        horizon = np.zeros(len(detectors))
        for j in range(len(detectors)):
            Fp = self.ifos[j].antenna_response(ra_, dec_, geocent_time_, psi_, "plus")
            Fc = self.ifos[j].antenna_response(ra_, dec_, geocent_time_, psi_, "cross")
            Deff2 = luminosity_distance_ / np.sqrt(
                Fp**2 * ((1 + np.cos(iota_) ** 2) / 2) ** 2
                + Fc**2 * np.cos(iota_) ** 2
            )

            horizon[j] = (Deff2 / snr_threshold) * opt_snr_unscaled[detectors[j]]

        return horizon

    ####################################################
    #                                                  #
    #   half_snr vs mtot table for interpolation       #
    #                                                  #
    ####################################################
    def init_halfScaled(self):
        """

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        mtot_min = self.mtot_min
        mtot_max = self.mtot_max
        nsamples = self.nsamples
        detectors = self.list_of_detectors.copy()
        print(f"Generating interpolator for {detectors} detectors")

        try:
            if mtot_min < 1.0:
                raise ValueError
        except ValueError:
            print("Error: mass too low")

        f_min = self.f_min

        # geocent_time cannot be array here
        # this geocent_time is only to get halfScaledSNR
        geocent_time_ = 1246527224.169434  # random time from O3

        iota_, ra_, dec_, psi_, phase_ = 0.0, 0.0, 0.0, 0.0, 0.0
        luminosity_distance_ = 100.0

        ratio = self.ratio
        snrHalf_ = np.zeros((len(ratio), len(detectors)), dtype=object)
        i = 0
        for q in tqdm(
            ratio,
            desc="interpolation for each mass_ratios",
            total=len(ratio),
            ncols=100,
        ):
            mass_ratio = q
            if self.waveform_inspiral_must_be_above_fmin == True:
                func = lambda x: self.findchirp_chirptime(
                    x / (1 + mass_ratio), x / (1 + mass_ratio) * mass_ratio, f_min
                )
                mtot_max = fsolve(func, 150)[
                    0
                ]  # to make sure that chirptime is not negative, TaylorF2 might need this

            # mtot_table = np.linspace(mtot_min,  mtot_max, nsamples)
            mtot_table = np.sort(
                mtot_min + mtot_max - np.geomspace(mtot_min, mtot_max, nsamples)
            )
            # mtot_table = np.sort(mtot_min+mtot_max-np.geomspace(mtot_min,  mtot_max, nsamples))
            mass_1_ = mtot_table / (1 + q)
            mass_2_ = mass_1_ * q
            mchirp = ((mass_1_ * mass_2_) ** (3 / 5)) / ((mtot_table) ** (1 / 5))
            ######## calling bilby_snr ########
            opt_snr_unscaled = self.compute_bilby_snr_(
                mass_1=mass_1_,
                mass_2=mass_2_,
                luminosity_distance=luminosity_distance_,
                theta_jn=iota_,
                psi=psi_,
                ra=ra_,
                dec=dec_,
                verbose=False,
                jsonFile=False,
            )

            A2 = mchirp ** (5.0 / 6.0)
            ######## filling in interpolation table for different detectors ########
            for j in range(len(detectors)):
                Fp = self.ifos[j].antenna_response(
                    ra_, dec_, geocent_time_, psi_, "plus"
                )
                Fc = self.ifos[j].antenna_response(
                    ra_, dec_, geocent_time_, psi_, "cross"
                )
                Deff2 = luminosity_distance_ / np.sqrt(
                    Fp**2 * ((1 + np.cos(iota_) ** 2) / 2) ** 2
                    + Fc**2 * np.cos(iota_) ** 2
                )

                snrHalf_[i, j] = interp1d(
                    mtot_table,
                    (Deff2 / A2) * opt_snr_unscaled[detectors[j]],
                    kind="cubic",
                )

            i += 1  # iterator over mass_ratio

        # save the interpolators for each detectors
        for j in range(len(detectors)):
            with open(self.interpolator_dict[detectors[j]], "wb") as handle:
                pickle.dump(snrHalf_[:, j], handle, protocol=pickle.HIGHEST_PROTOCOL)

        return None

    ####################################################
    #                                                  #
    #                    bilby snr                     #
    #                                                  #
    ####################################################
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
        psds=False,
        psd_file=True,
        psd_with_time=False,
        verbose=True,
        jsonFile=False,
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
        psds : dict, optional
            psd dict. if set False will get the values set at class initialization
            example_1=> when values are psd name from pycbc analytical psds,
            psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044'}
            example_2=> when values are psd txt file in bilby or custom created,
            psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt'}
            custom created txt file has two columns. 1st column: frequency array, 2nd column: strain
        psd_file : bool, optional
            if set True, the given value of psds param should be of psds instead of asd. If asd, set psd_file=False.
            e.g. if psd is {'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_early_high_psd.txt'}, then psd_file = [False,True].
        psd_with_time : float, optional
            gps end time end strain data for which psd will be found. (this param will be given highest priority)
            example=> psd_with_time=1246527224.169434
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
            example of opt_snr_unscaled return values for len(mass_1)=3
            {'opt_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
            'L1': array([132.08275995, 205.04492349, 246.47822334]),
            'H1': array([ 84.00372897, 130.40716432, 156.75845871])}
        """

        npool = self.npool
        geocent_time_ = 1246527224.169434  # random time from O3
        sampling_frequency = self.sampling_frequency
        if psds == False:
            # if psds are not given at the method level get what whatever that was initialized at the class level
            detectors = self.list_of_detectors
            psd_file = self.psd_file
            psds = self.psds.copy()
        else:
            # if psds are given at the method level
            detectors = list(psds.keys())
        approximant = self.waveform_approximant
        f_min = self.f_min

        ################
        # psd handling #
        ################
        # if psds information is not manually given, we will use the one provided in bilby for O3 sensitivity
        psds_arrays = dict()
        #######################################
        # more realistic psds from averaging a segment of gwosc data
        # psd calculation from gps time point
        # add exception handling for unrecognised time
        if psd_with_time != False:
            print("wait for sometime while psd data is being fetch...")
            # Use gwpy to fetch the open data
            duration = 16.0
            roll_off = 0.2
            psd_duration = duration * 32.0  # uint (seconds)
            psd_start_time = psd_with_time - psd_duration
            for ifo in detectors:
                psd_data = TimeSeries.fetch_open_data(
                    ifo,
                    psd_start_time,
                    psd_start_time + psd_duration,
                    sample_rate=sampling_frequency,
                    cache=True,
                )

                psd_alpha = 2 * roll_off / duration
                det_psd = psd_data.psd(
                    fftlength=duration,
                    overlap=0.5,
                    window=("tukey", psd_alpha),
                    method="median",
                )

                psds_arrays[ifo] = bilby.gw.detector.PowerSpectralDensity(
                    frequency_array=det_psd.frequencies.value, psd_array=det_psd.value
                )
        # get psd with psd name or with txt file
        else:
            i = 0  # iterator
            for det in detectors:
                # either provided psd or what's available in bilby
                if type(psds[det]) == str and psds[det][-3:] == "txt":
                    if psd_file[i]:
                        psds_arrays[det] = self.power_spectral_density(psds[det])
                    else:
                        psds_arrays[det] = self.amplitude_spectral_density(psds[det])

                # name string should be avaible in the list of psds in pycbc
                elif type(psds[det]) == str:
                    psds_arrays[det] = self.power_spectral_density_pycbc(psds[det])

                else:
                    raise Exception(
                        "the psds format is not recognised. The parameter psds dict should contain chosen detector names as keys \
                            and corresponding psds txt file name (or name from pycbc psd)as their values'"
                    )
                i += 1  # iterator wrt detectors

        #######################################
        # check whether there is input for geocent_time
        if not np.array(geocent_time).tolist():
            geocent_time = geocent_time_

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

        iter_ = []
        SNRs_list = []
        SNRs_dict = {}
        # time duration calculation for each of the mass combination
        safety = 1.2
        approx_duration = safety * self.findchirp_chirptime(mass_1, mass_2, f_min)
        duration = np.ceil(approx_duration + 2.0)

        if self.waveform_inspiral_must_be_above_fmin == True:
            # select only those that have inspiral part above f_min
            idx = approx_duration > 0.0

            # setting up parameters for feeding the inner product calculator (multiprocessing)
            size1 = len(mass_1)
            size2 = len(
                mass_1[idx]
            )  # chossing only those that have inspiral part above f_min
            iterations = np.arange(size1)  # to keep track of index
            iterations = iterations[idx]  # to keep track of index

            dectectorList = np.array(detectors) * np.ones(
                (size2, len(detectors)), dtype=object
            )
            psds_arrays_list = np.array([np.full(size2, psds_arrays, dtype=object)]).T

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
                    np.full(size2, approximant),
                    np.full(size2, f_min),
                    duration[idx],
                    np.full(size2, sampling_frequency),
                    iterations,
                ],
                dtype=object,
            ).T
        else:
            # setting up parameters for feeding the inner product calculator (multiprocessing)
            size1 = len(mass_1)
            iterations = np.arange(size1)  # to keep track of index

            dectectorList = np.array(detectors) * np.ones(
                (size1, len(detectors)), dtype=object
            )
            psds_arrays_list = np.array([np.full(size1, psds_arrays, dtype=object)]).T

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
            (input_arguments, psds_arrays_list, dectectorList), axis=1
        )

        #######################################
        # if inspiral only waveform, apply fmax cutoff
        if self.waveform_type == "Inspiral":
            with Pool(processes=npool) as pool:
                # call the same function with different data in parallel
                # imap->retain order in the list, while map->doesn't
                for result in tqdm(
                    pool.imap_unordered(self.snr_with_fmax_cutoff, input_arguments),
                    total=len(input_arguments),
                    ncols=100,
                    disable=not verbose,
                ):
                    iter_.append(result[1])
                    SNRs_list.append(result[0])
        else:
            with Pool(processes=npool) as pool:
                # call the same function with different data in parallel
                # imap->retain order in the list, while map->doesn't
                for result in tqdm(
                    pool.imap_unordered(
                        self.noise_weighted_inner_prod, input_arguments
                    ),
                    total=len(input_arguments),
                    ncols=100,
                    disable=not verbose,
                ):
                    iter_.append(result[1])
                    SNRs_list.append(result[0])
        #######################################

        # to fill in the snr values at the right index
        SNRs_list = np.array(SNRs_list)
        i = 0
        for det in detectors:
            snrs_ = np.zeros(size1)
            snrs_[iter_] = SNRs_list[:, i]
            SNRs_dict[det] = snrs_
            i = i + 1

        snrs_ = np.zeros(size1)
        snrs_[iter_] = SNRs_list[:, i]
        SNRs_dict["opt_snr_net"] = snrs_
        self.stored_snrs = SNRs_dict  # this stored snrs can be use for Pdet calculation

        # saving as json file
        if jsonFile:
            parameters_dict = {
                "mass_1": mass_1,
                "mass_2": mass_2,
                "luminosity_distance": luminosity_distance,
                "theta_jn": theta_jn,
                "psi": psi,
                "phase": phase,
                "ra": ra,
                "dec": dec,
                "geocent_time": geocent_time,
                "a_1": a_1,
                "a_2": a_2,
                "tilt_1": tilt_1,
                "tilt_2": tilt_2,
                "phi_12": phi_12,
                "phi_jl": phi_jl,
            }
            parameters_dict.update(SNRs_dict)
            file_name = "./bilby_GWparams_innerproductSNRs.json"
            json_dump = json.dumps(parameters_dict, cls=NumpyEncoder)
            with open(file_name, "w") as write_file:
                json.dump(json.loads(json_dump), write_file, indent=4)

        # how to load data form .json file
        # f = open ('data.json', "r")
        # data = json.loads(f.read())
        return SNRs_dict

    ####################################################
    #                                                  #
    #            psd array finder from bilby           #
    #                                                  #
    ####################################################
    def power_spectral_density(self, psd):
        """
        psd array finder from bilby

        Parameters
        ----------
        psd : str
            name of the psd
            e.g. 'aLIGO_O4_high_psd.txt'

        Returns
        -------
        psd_array : bilby.gw.detector.psd.PowerSpectralDensity object
        """
        return bilby.gw.detector.PowerSpectralDensity(psd_file=psd)

    ####################################################
    #                                                  #
    #            asd array finder from bilby           #
    #                                                  #
    ####################################################
    def amplitude_spectral_density(self, asd):
        """
        asd array finder from bilby

        Parameters
        ----------
        asd : str
            name of the asd
            e.g. 'aLIGO_O4_high_asd.txt'

        Returns
        -------
        psd_array : bilby.gw.detector.psd.PowerSpectralDensity object
        """
        return bilby.gw.detector.PowerSpectralDensity(asd_file=asd)

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
        delta_f = 1.0 / 16.0
        flen = int(self.sampling_frequency / delta_f)
        low_frequency_cutoff = self.f_min
        psd_ = pycbc.psd.from_string(psd, flen, delta_f, low_frequency_cutoff)
        return bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd_.sample_frequencies, psd_array=psd_.data
        )

    ####################################################
    #                                                  #
    #     SNR with f_max cutoff (Multiprocessing)      #
    #      (needed for inspiral only waveforms)        #
    #                                                  #
    ####################################################
    def snr_with_fmax_cutoff(self, params):
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

        f_min = params[16]
        f_max = (C**3) / (
            G * (params[0] + params[1]) * Mo * np.pi * 6 ** (3 / 2)
        )  # last stable orbit frequency
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

        # f_max for for cutoff
        f_array = waveform_generator.frequency_array
        idx = (f_array >= f_min) & (f_array <= f_max)
        h_plus = polas["plus"][idx]
        h_cross = polas["cross"][idx]

        SNRs_list = []
        NetSNR = 0.0
        # detectors = self.list_of_detectors
        list_of_detectors = params[21:].tolist()
        psds_arrays = params[20]
        for i in range(len(list_of_detectors)):
            # need to compute the inner product for
            p_array = psds_arrays[
                list_of_detectors[i]
            ].get_power_spectral_density_array(f_array)[idx]
            idx2 = (p_array != 0.0) & (p_array != np.inf)
            hp_inner_hp = bilby.gw.utils.noise_weighted_inner_product(
                h_plus[idx2], h_plus[idx2], p_array[idx2], waveform_generator.duration
            )
            hc_inner_hc = bilby.gw.utils.noise_weighted_inner_product(
                h_cross[idx2], h_cross[idx2], p_array[idx2], waveform_generator.duration
            )
            # make an ifo object to get the antenna pattern
            Fp = self.ifos[i].antenna_response(
                parameters["ra"],
                parameters["dec"],
                parameters["geocent_time"],
                parameters["psi"],
                "plus",
            )
            Fc = self.ifos[i].antenna_response(
                parameters["ra"],
                parameters["dec"],
                parameters["geocent_time"],
                parameters["psi"],
                "cross",
            )

            snrs_sq = abs((Fp**2) * hp_inner_hp + (Fc**2) * hc_inner_hc)

            SNRs_list.append(np.sqrt(snrs_sq))
            NetSNR += snrs_sq

        SNRs_list.append(np.sqrt(NetSNR))

        return (SNRs_list, params[19])

    ####################################################
    #                                                  #
    #  Noise weigthed inner product (Multiprocessing)  #
    #                                                  #
    ####################################################
    def noise_weighted_inner_prod(self, params):
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
        # <h|h>
        # 

        SNRs_list = []
        NetSNR = 0.0
        list_of_detectors = params[21:].tolist()
        psds_arrays = params[20]
        for i in range(len(list_of_detectors)):
            # need to compute the inner product for
            p_array = psds_arrays[
                list_of_detectors[i]
            ].get_power_spectral_density_array(waveform_generator.frequency_array)
            idx2 = (p_array != 0.0) & (p_array != np.inf)
            hp_inner_hp = bilby.gw.utils.noise_weighted_inner_product(
                polas["plus"][idx2],
                polas["plus"][idx2],
                p_array[idx2],
                waveform_generator.duration,
            )
            hc_inner_hc = bilby.gw.utils.noise_weighted_inner_product(
                polas["cross"][idx2],
                polas["cross"][idx2],
                p_array[idx2],
                waveform_generator.duration,
            )
            """
            hp_inner_hc = bilby.gw.utils.noise_weighted_inner_product(
                polas["plus"][idx2],
                polas["cross"][idx2],
                p_array[idx2],
                waveform_generator.duration,
            )
            """
            # make an ifo object to get the antenna pattern
            Fp = self.ifos[i].antenna_response(
                parameters["ra"],
                parameters["dec"],
                parameters["geocent_time"],
                parameters["psi"],
                "plus",
            )
            Fc = self.ifos[i].antenna_response(
                parameters["ra"],
                parameters["dec"],
                parameters["geocent_time"],
                parameters["psi"],
                "cross",
            )
            """
            snrs_sq = abs(
                (Fp**2) * hp_inner_hp
                + (Fc**2) * hc_inner_hc
                + 2 * Fp * Fc * hp_inner_hc
            )
            """
            snrs_sq = abs((Fp**2) * hp_inner_hp + (Fc**2) * hc_inner_hc)

            SNRs_list.append(np.sqrt(snrs_sq))
            NetSNR += snrs_sq

        SNRs_list.append(np.sqrt(NetSNR))

        return (SNRs_list, params[19])

    ####################################################
    #                                                  #
    #             Probaility of detection              #
    #                                                  #
    ####################################################
    def pdet(self, snrs=False, rho_th=8.0, rhoNet_th=8.0):
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
        if snrs == False:
            snrs = self.stored_snrs

        detectors = self.list_of_detectors
        pdet_dict = {}
        for det in detectors:
            pdet_dict["pdet_" + det] = 1 - norm.cdf(rho_th - snrs[det])

        pdet_dict["pdet_net"] = 1 - norm.cdf(rhoNet_th - snrs["opt_snr_net"])

        return pdet_dict

    ####################################################
    #                                                  #
    #                   Chirp time                     #
    #                                                  #
    ####################################################
    def findchirp_chirptime(self, m1, m2, fmin=20.0):
        """
        Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.
        -----------------
        Input parameters
        -----------------
        m1         : component mass of BBH, m1>m2, unit(Mo)
        m2         : component mass of BBH, m1>m2, unit(Mo)
        fmin       : minimum frequency cut-off for the analysis, unit(s)
        -----------------
        Return values
        -----------------
        chirp_time : Time taken from f_min to f_lso (frequency at last stable orbit), unit(s)
        """
        # variables used to compute chirp time
        m = m1 + m2
        eta = m1 * m2 / m / m
        c0T = c2T = c3T = c4T = c5T = c6T = c6LogT = c7T = 0.0

        c7T = Pi * (
            14809.0 * eta * eta / 378.0 - 75703.0 * eta / 756.0 - 15419335.0 / 127008.0
        )

        c6T = (
            Gamma * 6848.0 / 105.0
            - 10052469856691.0 / 23471078400.0
            + Pi * Pi * 128.0 / 3.0
            + eta * (3147553127.0 / 3048192.0 - Pi * Pi * 451.0 / 12.0)
            - eta * eta * 15211.0 / 1728.0
            + eta * eta * eta * 25565.0 / 1296.0
            + eta * eta * eta * 25565.0 / 1296.0
            + np.log(4.0) * 6848.0 / 105.0
        )
        c6LogT = 6848.0 / 105.0

        c5T = 13.0 * Pi * eta / 3.0 - 7729.0 * Pi / 252.0

        c4T = 3058673.0 / 508032.0 + eta * (5429.0 / 504.0 + eta * 617.0 / 72.0)
        c3T = -32.0 * Pi / 5.0
        c2T = 743.0 / 252.0 + eta * 11.0 / 3.0
        c0T = 5.0 * m * MTSUN_SI / (256.0 * eta)

        # This is the PN parameter v evaluated at the lower freq. cutoff
        xT = pow(Pi * m * MTSUN_SI * fmin, 1.0 / 3.0)
        x2T = xT * xT
        x3T = xT * x2T
        x4T = x2T * x2T
        x5T = x2T * x3T
        x6T = x3T * x3T
        x7T = x3T * x4T
        x8T = x4T * x4T

        # Computes the chirp time as tC = t(v_low)
        # tC = t(v_low) - t(v_upper) would be more
        # correct, but the difference is negligble.
        return (
            c0T
            * (
                1
                + c2T * x2T
                + c3T * x3T
                + c4T * x4T
                + c5T * x5T
                + (c6T + c6LogT * np.log(xT)) * x6T
                + c7T * x7T
            )
            / x8T
        )


# Store as JSON a numpy.ndarray or any nested-list composition.
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
