# -*- coding: utf-8 -*-
"""
Helper functions for gwsnr
"""

import os
import json
# supress warning
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from importlib import resources
import pickle
import numpy as np
import bilby
from gwpy.timeseries import TimeSeries

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy data types. It converts numpy.ndarray objects (and any nested-list composition
    that includes ndarray objects) into regular lists for JSON serialization. This is particularly useful when
    serializing data structures that include numpy arrays.
    """

    def default(self, obj):
        # Check if the object is an instance of numpy.ndarray
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        # Fallback to default behavior for other types
        return super(NumpyEncoder, self).default(obj)

def save_json(file_name, param):
    """Save a dictionary as a json file.

    Parameters
    ----------
    file_name : `str`
        json file name for storing the parameters.
    param : `dict`
        dictionary to be saved as a json file.
    """
    with open(file_name, "w", encoding="utf-8") as write_file:
        try:
            json.dump(param, write_file)
        except:
            json.dump(param, write_file, indent=4, cls=NumpyEncoder)

def load_json(file_name):
    """Load a json file.

    Parameters
    ----------
    file_name : `str`
        json file name for storing the parameters.

    Returns
    ----------
    param : `dict`
    """
    with open(file_name, "r", encoding="utf-8") as f:
        param = json.load(f)

    return param

def save_pickle(file_name, param):
    """Save a dictionary as a pickle file.

    Parameters
    ----------
    file_name : `str`
        pickle file name for storing the parameters.
    param : `dict`
        dictionary to be saved as a pickle file.
    """
    with open(file_name, "wb") as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    """Load a pickle file.

    Parameters
    ----------
    file_name : `str`
        pickle file name for storing the parameters.

    Returns
    ----------
    param : `dict`
    """
    with open(file_name, "rb") as f:
        param = pickle.load(f)

    return param

def load_ann_h5(filename):
    """
    Function to load a specific dataset from an .h5 file

    Parameters
    ----------
    filename : str
        name of the .h5 file
        
    Returns
    ----------
    model : `keras.models.Model`
        Keras model loaded from the .h5 file
    """

    return load_model(filename)

def append_json(file_name, new_dictionary, old_dictionary=None, replace=False):
    """
    Append (values with corresponding keys) and update a json file with a dictionary. There are four options:

    1. If old_dictionary is provided, the values of the new dictionary will be appended to the old dictionary and save in the 'file_name' json file.
    2. If replace is True, replace the json file (with the 'file_name') content with the new_dictionary.
    3. If the file does not exist, create a new one with the new_dictionary.
    4. If none of the above, append the new dictionary to the content of the json file.

    Parameters
    ----------
    file_name : `str`
        json file name for storing the parameters. 
    new_dictionary : `dict`
        dictionary to be appended to the json file.
    old_dictionary : `dict`, optional
        If provided the values of the new dictionary will be appended to the old dictionary and save in the 'file_name' json file. 
        Default is None.
    replace : `bool`, optional
        If True, replace the json file with the dictionary. Default is False.

    """

    # check if the file exists
    # time
    # start = datetime.datetime.now()
    if old_dictionary:
        data = old_dictionary
    elif replace:
        data = new_dictionary
    elif not os.path.exists(file_name):
        #print(f" {file_name} file does not exist. Creating a new one...")
        replace = True
        data = new_dictionary
    else:
        #print("getting data from file")
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    # end = datetime.datetime.now()
    # print(f"Time taken to load the json file: {end-start}")

    # start = datetime.datetime.now()
    if not replace:
        data = add_dictionaries_together(data, new_dictionary)
        # data_key = data.keys()
        # for key, value in new_dictionary.items():
        #     if key in data_key:
        #         data[key] = np.concatenate((data[key], value)).tolist()
    # end = datetime.datetime.now()
    # print(f"Time taken to append the dictionary: {end-start}")

    # save the dictionary
    # start = datetime.datetime.now()
    #print(data)
    with open(file_name, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, indent=4, cls=NumpyEncoder)
    # end = datetime.datetime.now()
    # print(f"Time taken to save the json file: {end-start}")

    return data

def add_dictionaries_together(dictionary1, dictionary2):
    """
    Adds two dictionaries with the same keys together.
    
    Parameters
    ----------
    dictionary1 : `dict`
        dictionary to be added.
    dictionary2 : `dict`
        dictionary to be added.

    Returns
    ----------
    dictionary : `dict`
        dictionary with added values.
    """
    dictionary = {}
    # Check if either dictionary empty, in which case only return the dictionary with values
    if len(dictionary1) == 0:
        return dictionary2
    elif len(dictionary2) == 0:
        return dictionary1
    # Check if the keys are the same
    if dictionary1.keys() != dictionary2.keys():
        raise ValueError("The dictionaries have different keys.")
    for key in dictionary1.keys():
        value1 = dictionary1[key]
        value2 = dictionary2[key]

        # check if the value is empty
        bool0 = len(value1) == 0 or len(value2) == 0
        # check if the value is an ndarray or a list
        bool1 = isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray)
        bool2 = isinstance(value1, list) and isinstance(value2, list)
        bool3 = isinstance(value1, np.ndarray) and isinstance(value2, list)
        bool4 = isinstance(value1, list) and isinstance(value2, np.ndarray)
        bool4 = bool4 or bool3
        bool5 = isinstance(value1, dict) and isinstance(value2, dict)

        if bool0:
            if len(value1) == 0 and len(value2) == 0:
                dictionary[key] = np.array([])
            elif len(value1) != 0 and len(value2) == 0:
                dictionary[key] = np.array(value1)
            elif len(value1) == 0 and len(value2) != 0:
                dictionary[key] = np.array(value2)
        elif bool1:
            dictionary[key] = np.concatenate((value1, value2))
        elif bool2:
            dictionary[key] = value1 + value2
        elif bool4:
            dictionary[key] = np.concatenate((np.array(value1), np.array(value2)))
        elif bool5:
            dictionary[key] = add_dictionaries_together(
                dictionary1[key], dictionary2[key]
            )
        else:
            raise ValueError(
                "The dictionary contains an item which is neither an ndarray nor a dictionary."
            )
    return dictionary

def get_param_from_json(json_file):
    """
    Function to get the parameters from json file.

    Parameters
    ----------
    json_file : `str`
        json file name for storing the parameters.

    Returns
    ----------
    param : `dict`
    """
    with open(json_file, "r", encoding="utf-8") as f:
        param = json.load(f)

    for key, value in param.items():
        param[key] = np.array(value)
    return param

def load_ann_h5_from_module(package, directory, filename):
    """
    Function to load a specific dataset from an .h5 file within the package

    Parameters
    ----------
    package : str
        name of the package
    directory : str
        name of the directory within the package
    filename : str
        name of the .h5 file

    Returns
    ----------
    model : `keras.models.Model`
        Keras model loaded from the .h5 file
    """

    with resources.path(package + '.' + directory, filename) as h5_path:
        return load_model(h5_path)

def load_json_from_module(package, directory, filename):
    """
    Function to load a specific dataset from a .json file within the package

    Parameters
    ----------
    package : str
        name of the package
    directory : str
        name of the directory within the package
    filename : str
        name of the .json file

    Returns
    ----------
    data : `dict`
        Dictionary loaded from the .json file
    """

    with resources.path(package + '.' + directory, filename) as json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
def load_pickle_from_module(package, directory, filename):
    """
    Function to load a specific dataset from a .pkl file within the package

    Parameters
    ----------
    package : str
        name of the package
    directory : str
        name of the directory within the package
    filename : str
        name of the .pkl file

    Returns
    ----------
    data : `dict`
        Dictionary loaded from the .pkl file
    """
    
    with resources.path(package + '.' + directory, filename) as pkl_path:
        return pickle.load(open(pkl_path, "rb"))

def dealing_with_psds(psds=None, ifos=None, f_min=20.0, sampling_frequency=2048.0):
    """
    Function to deal with psds inputs and for creating bilby.gw.detector.PowerSpectralDensity objects.

    Parameters
    ----------
    psds : dict
        dictionary of psds. psds.keys()=detector names, psds.values()=psds file names or pycbc psd names
    ifos : `list` or `None`
        List of interferometer objects or interferometer name list. Default is None. If None, bilby's default interferometer objects will be used.
    f_min : `float`
        Minimum frequency of the psds. Default is 20.
    sampling_frequency : `float`
        Sampling frequency of the psds. Default is 2048.

    Returns
    ----------
    psds_list : `list`
        list of bilby.gw.detector.PowerSpectralDensity objects
    detector_tensor_list : `list`
        list of detector tensors
    detector_list : `list`
        list of detector names
    """

    if not psds and not ifos:
        # if psds is not given, choose bilby's default psds
        print("psds not given. Choosing bilby's default psds")
        psds = dict()
        psds["L1"] = "aLIGO_O4_high_asd.txt"
        psds["H1"] = "aLIGO_O4_high_asd.txt"
        psds["V1"] = "AdV_asd.txt"
        detector_list = list(psds.keys())
        # for Fp, Fc calculation
        ifos = bilby.gw.detector.InterferometerList(detector_list)

    elif ifos and not psds:
        ifos_ = []
        detector_list = []
        psds = dict()
        for ifo in ifos:
            if isinstance(ifo, str):
                if ifo == "ET":
                    ifos_ += bilby.gw.detector.InterferometerList([ifo])
                    detector_list.append("ET1")
                    detector_list.append("ET2")
                    detector_list.append("ET3")
                    psds["ET1"] = ifos_[-3].power_spectral_density.psd_file
                    psds["ET2"] = ifos_[-2].power_spectral_density.psd_file
                    psds["ET3"] = ifos_[-1].power_spectral_density.psd_file
                else:
                    ifos_.append(bilby.gw.detector.InterferometerList([ifo])[0])
                    detector_list.append(ifo)
                    psds[ifo] = ifos_[-1].power_spectral_density.psd_file
                    if not psds[ifo]:
                        psds[ifo] = ifos_[-1].power_spectral_density.asd_file

            else:
                ifos_.append(ifo)
                detector_list.append(ifo.name)
                psds[ifo.name] = ifo.power_spectral_density.psd_file
                if not psds[ifo.name]:
                    psds[ifo.name] = ifo.power_spectral_density.asd_file
        ifos = ifos_

    elif psds and not ifos:
        detector_list = list(psds.keys())
        ifos = bilby.gw.detector.InterferometerList(detector_list)

        if "ET" in detector_list:
            # insert ET1, ET2, ET3 inplace of ET in the detector_list
            idx = detector_list.index("ET")
            detector_list.pop(idx)
            detector_list.insert(idx, "ET1")
            detector_list.insert(idx + 1, "ET2")
            detector_list.insert(idx + 2, "ET3")

        # for i, ifo in enumerate(ifos):
        #     psds[ifo.name] = ifo.power_spectral_density.psd_file

    elif psds and ifos:
        detector_list = []
        ifos_ = []
        psds_ = dict()
        for ifo in ifos:
            if isinstance(ifo, str):
                if ifo == "ET":
                    ifos_ += bilby.gw.detector.InterferometerList([ifo])
                    detector_list.append("ET1")
                    detector_list.append("ET2")
                    detector_list.append("ET3")
                    psds_["ET1"] = psds["ET"]
                    psds_["ET2"] = psds["ET"]
                    psds_["ET3"] = psds["ET"]
                else:
                    ifos_.append(bilby.gw.detector.InterferometerList([ifo])[0])
                    detector_list.append(ifo)
                    psds_[ifo] = psds[ifo]

            else:
                ifos_.append(ifo)
                detector_list.append(ifo.name)
                psds_[ifo.name] = psds[ifo.name]

        ifos = ifos_
        psds = psds_
    else:
        raise ValueError("psds and ifos are not in the correct format")

    # generate bilby's psd objects
    psds_list = []
    detector_tensor_list = []
    error_msg = "the psds format is not recognised. The parameter psds dict should contain chosen detector names as keys and corresponding psds txt file name (or name from pycbc psd) as their values"
    # print(psds)
    # print(detector_list)
    for i, det in enumerate(detector_list):
        # either provided psd or what's available in bilby

        if type(psds[det]) == str and psds[det][-3:] == "txt":

            if psds[det][-7:] == "psd.txt":
                psds_list.append(
                    bilby.gw.detector.PowerSpectralDensity(psd_file=psds[det])
                )

            elif psds[det][-7:] == "asd.txt":
                psds_list.append(
                    bilby.gw.detector.PowerSpectralDensity(asd_file=psds[det])
                )
            else:
                raise ValueError(
                    "psd file name should end with either 'psd.txt' or 'asd.txt'"
                )

        elif isinstance(psds[det], float):
            # get the psd from the open data
            duration = 16
            sample_rate = 4096.0  # 2048Hz throws error
            psd_duration = duration * 32
            analysis_start = psds[det]
            psd_start_time = analysis_start - psd_duration

            # check directory
            if not os.path.exists("./psd_data"):
                os.makedirs("./psd_data")

            # check if the txt file exists in psd directory
            path_ = f'./psd_data/{det}_{int(analysis_start+1)}_psd.txt'
            it_exist = os.path.exists(path_)

            if not it_exist:
                # set up empty interferometer
                X1 = bilby.gw.detector.get_empty_interferometer(det)

                # download the data
                print(f"Downloading data for {det} detector")
                X1_psd_data = TimeSeries.fetch_open_data(
                    det, psd_start_time, psd_start_time + psd_duration, sample_rate=sample_rate, cache=True)

                # calculate the psd
                psd_alpha = 2 * X1.strain_data.roll_off / duration
                X1_psd = X1_psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")

                # save the psd
                print(f"Saving psd data for {det} detector at {path_}")
                X1_psd.write(path_)

                # initialize the psd object
                psd_obj = bilby.gw.detector.PowerSpectralDensity(
                        frequency_array=X1_psd.frequencies.value, psd_array=X1_psd.value
                    )
            else:
                print(f"Loading psd data for {det} detector from {path_}")
                psd_obj = bilby.gw.detector.PowerSpectralDensity(psd_file=path_)

            psds_list.append(psd_obj)

        elif isinstance(psds[det], str):
            print("Trying to get the psd from pycbc: ", psds[det])
            psds_list.append(
                power_spectral_density_pycbc(psds[det], f_min, sampling_frequency)
            )
        else:
            raise ValueError(error_msg)
        
        detector_tensor_list.append(ifos[i].detector_tensor)

    return psds_list, detector_tensor_list, detector_list

def power_spectral_density_pycbc(psd, f_min=20.0, sampling_frequency=2048.0):
    """
    psd array finder from pycbc

    Parameters
    ----------
    psd : str
        name of the psd
        e.g. 'aLIGOaLIGODesignSensitivityT1800044'
    f_min : float
        minimum frequency of the psd
        default: 20.
    sampling_frequency : float
        sampling frequency of the psd
        default: 2048.

    Returns
    -------
    psd_array : bilby.gw.detector.psd.PowerSpectralDensity object
    """
    import pycbc
    import pycbc.psd

    delta_f = 1.0 / 16.0
    flen = int(sampling_frequency / delta_f)
    low_frequency_cutoff = f_min
    psd_ = pycbc.psd.from_string(psd, flen, delta_f, low_frequency_cutoff)
    return bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd_.sample_frequencies, psd_array=psd_.data
    )


# interpolator check and generation
def interpolator_check(
    param_dict_given,
    interpolator_dir,
    create_new,
):
    """
    Function for interpolator (snr_partialsacaled) check and generation if not exists.

    Parameters
    ----------
    param_dict_given : dict
        dictionary of parameters based on which the existence of interpolator will be checked
    interpolator_dir : str
        path to the interpolator pickle file
    create_new : bool
        if True, new interpolator will be generated even if the interpolator exists
        if False, existing interpolator will be used if exists, otherwise new interpolator will be generated

    Returns
    ----------
    psds_list_ : list
        list of psd objects
    detector_tensor_list_ : list
        list of detector tensors
    detector_list_ : list
        list of detector names
    """

    detector_list = param_dict_given["detector"]
    psds_list = param_dict_given["psds"]
    detector_tensor_list = param_dict_given["detector_tensor"]
    detector_list_ = []
    detector_tensor_list_ = []
    psds_list_ = []
    path_interpolator_ = []
    path_interpolator_all = []

    # getting interpolator if exists
    # for each detector, one by one
    k = 0
    for det in detector_list:
        param_dict_given["detector"] = det
        param_dict_given["psds"] = psds_list[k]
        param_dict_given["detector_tensor"] = detector_tensor_list[k]
        # checking 
        path_interpolator, it_exist = interpolator_pickle_path(
            param_dict_given, interpolator_dir
        )
        if create_new:
            it_exist = False
        if it_exist:
            print(
                f"Interpolator will be loaded for {det} detector from {path_interpolator}"
            )
        else:
            print(
                f"Interpolator will be generated for {det} detector at {path_interpolator}"
            )

            detector_list_.append(det)
            psds_list_.append(psds_list[k])
            detector_tensor_list_.append(detector_tensor_list[k])
            path_interpolator_.append(path_interpolator)

        path_interpolator_all.append(path_interpolator)
        k += 1

    return (
        psds_list_,
        detector_tensor_list_,
        detector_list_,
        path_interpolator_,
        path_interpolator_all,
    )


def interpolator_pickle_path(param_dict_given, path="./interpolator_pickle"):
    """
    Function for storing or getting interpolator (snr_partialsacaled) pickle path

    Parameters
    ----------
    param_dict_given : dict
        dictionary of parameters based on which the existence of interpolator will be checked
    path : str
        path to the directory where the interpolator pickle file will be stored

    Returns
    -------
    path_interpolator : str
        path to the interpolator pickle file
        e.g. './interpolator_pickle/L1/partialSNR_dict_0.pickle'
    it_exist: bool
        True if the interpolator exists
        False if the interpolator does not exists
    """

    detector = param_dict_given["detector"]
    # arg no. for the detector from the detector list
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
    param_dict_stored = pickle.load(open(det_path + "/param_dict_list.pickle", "rb"))

    len_ = len(param_dict_stored)
    # del param_dict_given["psds"]
    param_dict_given["detector_tensor"] = str(param_dict_given["detector_tensor"])
    # print("\n\n", param_dict_given)
    # print("\n\n",param_dict_stored)
    if param_dict_given in param_dict_stored:
        # try and except is added so that user can regenerate a new interpolator pickle file just by
        # deleting the right file and reruing gwsnr with that params again
        # also, if the user delete the file by mistake, it will generate in the next run
        idx = param_dict_stored.index(param_dict_given)
        # check if interpolator pickle exists
        # get partialSNR interpolator if exists
        path_interpolator = det_path + "/partialSNR_dict_" + str(idx) + ".pickle"
        # there will be exception if the file is deleted by mistake
        if os.path.exists(path_interpolator):
            it_exist = True
        else:
            it_exist = False

    # if related dict not found in the param_dict_list.pickle
    else:
        it_exist = False
        path_interpolator = det_path + "/partialSNR_dict_" + str(len_) + ".pickle"
        # print("related dict not found in the param_dict_list.pickle, new interpolator will be generated")

        # store the pickle dict
        param_dict_stored.append(param_dict_given)
        with open(det_path + "/param_dict_list.pickle", "wb") as handle:
            pickle.dump(param_dict_stored, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(f"In case if you need regeneration of interpolator of the given gwsnr param, please delete this file, {path_interpolator} \n")

    return (path_interpolator, it_exist)


def get_gw_parameters(gw_param_dict):

    if isinstance(gw_param_dict, dict):
        pass
    elif isinstance(gw_param_dict, str):
        try:
            gw_param_dict = get_param_from_json(gw_param_dict)
        except:
            raise ValueError("gw_param_dict should be either a dictionary or a json string")
    else:
        raise ValueError("gw_param_dict should be either a dictionary or a json string")

    mass_1 = gw_param_dict.get("mass_1", np.array([10.0]))
    mass_2 = gw_param_dict.get("mass_2", np.array([10.0]))
    mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
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

    # Extract tidal parameters or initialize to zeros
    lambda_1 = gw_param_dict.get("lambda_1", np.zeros(size))
    lambda_2 = gw_param_dict.get("lambda_2", np.zeros(size))
    
    # Extract eccentricity or initialize to zeros
    eccentricity = gw_param_dict.get("eccentricity", np.zeros(size))

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
        lambda_1,
        lambda_2,
        eccentricity,
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
        lambda_1,
        lambda_2,
        eccentricity,
    )

    return mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, eccentricity