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

    elif not psds and ifos:
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
    error_msg = "the psds format is not recognised. The parameter psds dict should contain chosen detector names as keys and corresponding psds txt file name (or name from pycbc psd)as their values"
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

            detector_tensor_list.append(ifos[i].detector_tensor)

        elif isinstance(psds[det], str):
            try:
                psds_list.append(
                    power_spectral_density_pycbc(psds[det]), f_min, sampling_frequency
                )
            except:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)

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