import numpy as np
import json
import bilby
import os
import pickle


class NumpyEncoder(json.JSONEncoder):
    """
    Store as JSON a numpy.ndarray or any nested-list composition.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def dealing_with_psds(psds=None, isit_psd_file=False, ifos=None):
    """
    Function to deal with psds.
    """
    
    if not psds:
        # if psds is not given, choose bilby's default psds
        print("psds not given. Choosing bilby's default psds")
        psds = dict()
        psds["L1"] = "aLIGO_O4_high_asd.txt"
        psds["H1"] = "aLIGO_O4_high_asd.txt"
        psds["V1"] = "AdV_asd.txt"
        detector_list = list(psds.keys())
        isit_psd_file_dict = dict(L1=False, H1=False, V1=False)
        # for Fp, Fc calculation
        ifos = bilby.gw.detector.InterferometerList(detector_list)
    else:
        # given psds is a dictionary
        # dict keys are the detectors names
        # dict values are the psds file names
        try:
            detector_list = list(psds.keys())
        except:
            print("psds must be a dictionary with keys as detector names (e.g.'L1' for Livingston etc) and values as str")
            raise ValueError
        
        if isinstance(isit_psd_file) == bool:  # this means all psds are 
            isit_psd_file_dict = dict()
            for det in detector_list:
                isit_psd_file_dict[det] = isit_psd_file
        elif list(isit_psd_file.keys())==detector_list:
            pass
        else:
            print(f"isit_psd_file must be a dictionary with keys: {detector_list} and values as bool")
            raise ValueError

        ifos = []
        len_ = len(detector_list)
        for i in range(len_):
            if ifos[i]:
                # if ifos is not None, then use the given ifos
                # ifos is a list of bilby.gw.detector.Interferometer 
                ifos.append(ifos[i])
            else:
                # if ifos is None, then generate bilby's default ifos with the given list of detectors
                ifos.append(
                    bilby.gw.detector.InterferometerList(
                        [detector_list[i]]
                    )[0]
                )

    # generate bilby's psd objects
    psds_list = []
    detector_tensor_list = []
    i = 0
    for det in detector_list:
        if isit_psd_file_dict[det]:
            psds_list.append(
                bilby.gw.detector.PowerSpectralDensity(
                    psd_file=psds[det]
                )
            )
        else:
            psds_list.append(
                bilby.gw.detector.PowerSpectralDensity(
                    asd_file=psds[det]
                )
            )
        detector_tensor_list.append(ifos[i].detector_tensor)
        i += 1

    return psds_list, detector_tensor_list, detector_list

# interpolator check and generation
def interpolator_check(
        param_dict_given, interpolator_dir, create_new,
):
    """
    Function for interpolator check and generation if not exists
    """

    detector_list = param_dict_given["detector"]
    psds_list = param_dict_given["psds"]
    detector_tensor_list = param_dict_given["detector_tensor"]
    detector_list_ = []
    detector_tensor_list_ = []
    psds_list_ = []
    path_interpolator_ = []

    # getting interpolator if exists
    k = 0
    for det in detector_list:
        param_dict_given["detector"] = det
        param_dict_given["psds"] = psds_list[k]
        param_dict_given["detector_tensor"] = detector_tensor_list[k]
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
        k += 1

    return psds_list_, detector_tensor_list_, detector_list_, path_interpolator_

def interpolator_pickle_path(
    param_dict_given, path="./interpolator_pickle"
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
    param_dict_stored = pickle.load(
        open(det_path + "/param_dict_list.pickle", "rb")
    )

    len_ = len(param_dict_stored)
    #del param_dict_given["psds"]
    param_dict_given["detector_tensor"] = str(param_dict_given["detector_tensor"])
    # print("\n\n", param_dict_given)
    # print("\n\n",param_dict_stored)
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
    with open(file_name, "rb") as handle:
        param = pickle.load(handle)

    return param

def save_json(param, file_name):
    """Save a json file.

    Parameters
    ----------
    param : `dict`
        dictionary of parameters.
    file_name : `str`
        json file name for storing the parameters.
    """
    with open(file_name, "wb") as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_json_dict(dict, file_name):
    """Save a json file.

    Parameters
    ----------
    param : `dict`
        dictionary of parameters.
    file_name : `str`
        json file name for storing the parameters.
    """

    json_dump = json.dumps(dict, cls=NumpyEncoder)
    with open(file_name, "w") as write_file:
        json.dump(json.loads(json_dump), write_file, indent=4)


def load_json_dict(file_name):
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