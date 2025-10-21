"""
This module implements the SNRThresholdFinder class to determine the optimal SNR threshold for gravitational wave detection using cross-entropy maximization (following Essick et al. 2023).
"""
import multiprocessing as mpp
from tqdm import tqdm
import h5py
import numpy as np
from .crossentropydifference import cross_entropy_difference
# import crossentropydifference
# cross_entropy_difference = crossentropydifference.cross_entropy_difference
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


class SNRThresholdFinder:
    """
    A class to find the optimal SNR threshold for gravitational wave detection using cross-entropy maximization.

    Parameters
    ----------
    catalog_file : str
        Path to the HDF5 file containing the injection catalog data. The file should have something like the following structure (refer to https://zenodo.org/records/16740117):
        ```
        injections.hdf
        |-- events
        |   |-- z  (parameter to me fitted on)
        |   |-- mass1_source (parameter with which the data is to be selected with)
        |   |-- gstlal_far (original_detection_statistic)
        |   |-- observed_snr_net (projected_detection_statistic)
        ```
    original_detection_statistic : dict, optional
        Dictionary specifying the original detection statistic with keys:
        'parameter' (str): Name of the key in the catalog for the original detection statistic.
        'threshold' (float): Threshold value for the original detection statistic.
        Default is {'parameter': 'gstlal_far', 'threshold': 1}.
    projected_detection_statistic : dict, optional
        Dictionary specifying the projected detection statistic with keys: 
        'parameter' (str): Name of the key in the catalog for the projected detection statistic.
        'threshold' (float): Threshold value for the projected detection statistic.
        'threshold_search_bounds' (tuple): Bounds for the threshold search.
        Default is {'parameter': 'observed_snr_net', 'threshold': None, 'threshold_search_bounds': (4, 14)}.
    parameters_to_fit : list of str, optional
        List of parameter to fit, e.g., ['redshift']. Default is ['redshift'].
    sample_size : int, optional
        Number of samples to use for KDE estimation. Default is 10000.
    selection_range : dict, optional
        Dictionary specifying the selection range with keys:
        'parameter' (str or list): Parameter(s) to apply the selection range on.
        'range' (tuple): Tuple specifying the (min, max) range for selection.
        Default is {'parameter': 'mass1_source', 'range': (5, 200)}.

    Examples
    ----------
    >>> finder = SNRThresholdFinder(catalog_file='injection_catalog.h5')
    >>> best_thr, del_H, H, H_true, snr_thrs = finder.find_threshold(iteration=10)
    >>> print(f"Best SNR threshold: {best_thr:.2f}")

    Instance Attributes
    ----------
    SNRThresholdFinder class has the following attributes, \n
    +-------------------------------------+----------------------------------+
    | Atrributes                          | Type                             |
    +=====================================+==================================+
    | original_detection_statistic       | dict                             |
    +-------------------------------------+----------------------------------+
    | projected_detection_statistic      | dict                             |
    +-------------------------------------+----------------------------------+
    | parameters_to_fit                 | list                             |
    +-------------------------------------+----------------------------------+
    | sample_size                        | int                              |
    +-------------------------------------+----------------------------------+
    | selection_range                    | dict                             |
    +-------------------------------------+----------------------------------+

    Instance Methods
    ----------
    GWSNR class has the following methods, \n
    +-------------------------------------+----------------------------------+
    | Methods                             | Description                      |
    +=====================================+==================================+
    | det_data                            | Load and preprocess catalog data |
    +-------------------------------------+----------------------------------+
    | find_threshold                      | Find the optimal SNR threshold   |
    +-------------------------------------+----------------------------------+
    | find_best_SNR_threshold             | Find the best SNR threshold using |
    |                                     | spline interpolation and         |
    |                                     | optimization                     |
    +-------------------------------------+----------------------------------+
    """

    def __init__(self, 
                catalog_file=None, 
                npool=4,
                multiprocessing_verbose=True,
                original_detection_statistic=None, projected_detection_statistic=None,
                parameters_to_fit=None,
                sample_size=20000,
                selection_range=None,
        ):

        self.npool = npool
        self.multiprocessing_verbose = multiprocessing_verbose

        if selection_range is None:
            selection_range = dict(
                key_name = 'mass1_source',
                parameter = None,
                range = (30, 60),
            )
        self.selection_range = selection_range

        if original_detection_statistic is None:
            self.original_detection_statistic = dict(
                key_name='gstlal_far',
                parameter=None,
                threshold=1,  # 1 per year
            )
        else:
            self.original_detection_statistic = original_detection_statistic

        if projected_detection_statistic is None:
            self.projected_detection_statistic = dict(
                key_name='observed_snr_net',
                parameter=None,
                threshold=None, # to be determined
                threshold_search_bounds=(4, 14),
            )
        else:
            self.projected_detection_statistic = projected_detection_statistic

        if parameters_to_fit is None:
            self.parameters_to_fit = dict(
                key_name = 'z',
                parameter = None,
            )
        else:
            self.parameters_to_fit = parameters_to_fit

        self.sample_size = sample_size

        self.det_data(catalog_file);

    def det_data(self, 
            catalog_file,
        ):
        """
        Function to load and preprocess the injection catalog data from an HDF5 file.

        Parameters
        ----------
        catalog_file : str
            Path to the HDF5 file containing the injection catalog data.
        Returns
        -------
        result_dict : dict
            Dictionary containing the preprocessed data for the specified parameters and detection statistics.

        Raises
        ------
        ValueError
            If 'redshift' is not included in parameters_to_fit.
        """
        def raise_not_provided(param):
            raise ValueError(f"if catalog_file is not provided, you must provide {param} as list or numpy array.")
        
        if catalog_file is None:
            param = self.selection_range['parameter']
            if isinstance(param, list) or isinstance(param, np.ndarray):
                self.selection_range['parameter'] = np.array(param)
            else:
                raise_not_provided('selection_range["parameter"]')
            
            param = self.original_detection_statistic['parameter']
            if isinstance(param, list) or isinstance(param, np.ndarray):
                self.original_detection_statistic['parameter'] = np.array(param)
            else:
                raise_not_provided('original_detection_statistic["parameter"]')

            param = self.projected_detection_statistic['parameter']
            if isinstance(param, list) or isinstance(param, np.ndarray):
                self.projected_detection_statistic['parameter'] = np.array(param)
            else:
                raise_not_provided('projected_detection_statistic["parameter"]')

            param = self.parameters_to_fit['parameter']
            if isinstance(param, list) or isinstance(param, np.ndarray):
                self.parameters_to_fit['parameter'] = np.array(param)
            else:
                raise_not_provided('parameters_to_fit["parameter"]')

        else:
            with h5py.File(catalog_file, 'r') as obj:
                attrs = dict(obj.attrs.items())
                events = obj['events'][:]

            key_name = self.selection_range['key_name']
            if key_name in events.dtype.names:
                self.selection_range['parameter'] = events[key_name]
            else:
                print(f"[WARNING] {key_name} not found in the catalog. Using the parameter array of the same name if provided.")

            key_name = self.original_detection_statistic['key_name']
            self.original_detection_statistic['parameter'] = events[key_name]

            key_name = self.projected_detection_statistic['key_name']
            self.projected_detection_statistic['parameter'] = events[key_name]

            key_name = self.parameters_to_fit['key_name']
            if isinstance(key_name, list):
                param_array = []
                for i, kn in enumerate(key_name):
                    if isinstance(self.parameters_to_fit['parameter'], list) or isinstance(self.parameters_to_fit['parameter'], np.ndarray):
                        param_array.append(self.parameters_to_fit['parameter'][i])
                    elif kn in events.dtype.names:
                        param_array.append(events[kn])
                    else:
                        raise ValueError(f"{kn} not found in the catalog. Please provide the parameter array of the same name.")
                self.parameters_to_fit['parameter'] = np.array(param_array)
            else:
                self.parameters_to_fit['parameter'] = events[key_name]

        # select only events within the selection range
        min_val = self.selection_range['range'][0]
        max_val = self.selection_range['range'][1]
        param = self.selection_range['parameter']
        idx_ = (param >= min_val) & (param <= max_val)

        dim = len(self.parameters_to_fit['parameter'].shape)
        if dim < 2:
            self.parameters_to_fit['parameter'] = self.parameters_to_fit['parameter'][idx_]
        else:
            raise NotImplementedError("Selection range filtering for multi-dimensional parameters_to_fit is not implemented yet.")
            # param_array = []
            # for i in range(dim):
            #     param_array.append(self.parameters_to_fit['parameter'][i][idx_])
            # self.parameters_to_fit['parameter'] = np.array(param_array)
        self.original_detection_statistic['parameter'] = self.original_detection_statistic['parameter'][idx_]
        self.projected_detection_statistic['parameter'] = self.projected_detection_statistic['parameter'][idx_]
        

    def find_threshold(self, iteration=10, print_output=True, no_multiprocessing=False):
        """
        Function to find the optimal SNR threshold by maximizing the cross-entropy difference.

        Parameters
        ----------
        iteration : int, optional
            Number of iterations for threshold search. Default is 10.
        print_output : bool, optional
            Whether to print the best SNR threshold. Default is True.

        Returns
        -------
        best_thr : float
            The optimal SNR threshold that maximizes the cross-entropy difference.
        del_H : np.ndarray
            Array of cross-entropy differences for each threshold tested.
        H : np.ndarray
            Array of cross-entropy values for the KDE with cut.
        H_true : np.ndarray
            Array of cross-entropy values for the original KDE.
        snr_thrs : np.ndarray
            Array of SNR thresholds tested.
            
        Raises
        ------
        ValueError
            If the number of iterations is less than 1.

        """

        snr_thrs = np.linspace(
            self.projected_detection_statistic['threshold_search_bounds'][0],
            self.projected_detection_statistic['threshold_search_bounds'][1],
            iteration
        )
        iters = np.arange(iteration)

        sample_size = self.sample_size
        parameters_to_fit = self.parameters_to_fit.copy()
        original_detection_statistic = self.original_detection_statistic.copy()
        projected_detection_statistic = self.projected_detection_statistic.copy()

        # set-up inputs for multoprocessing
        input_args = [(
            snr_thr, 
            sample_size,
            np.array(original_detection_statistic['parameter']),
            np.array(original_detection_statistic['threshold']),
            np.array(projected_detection_statistic['parameter']),
            np.array(parameters_to_fit['parameter']),
            iters[i]
        ) for i, snr_thr in enumerate(snr_thrs)]

        input_args = np.array(input_args, dtype=object)

        # test with for loop first before using multiprocessing
        del_H = np.zeros(iteration)
        H = np.zeros(iteration)
        H_true = np.zeros(iteration)

        if no_multiprocessing:
            for args in tqdm(input_args, total=len(input_args), ncols=100):
                del_H_i, H_i, H_true_i, iter_i = cross_entropy_difference(args)

                

                del_H[iter_i] = del_H_i
                H[iter_i] = H_i
                H_true[iter_i] = H_true_i
        else:
            print("if multiprocessing get stuck, use no_multiprocessing=True")
            npool = self.npool
            with mpp.Pool(processes=npool) as pool:
                self._multiprocessing_error()
                if self.multiprocessing_verbose:
                    for result in tqdm(
                            pool.imap_unordered
                            (cross_entropy_difference, input_args),
                            total=len(input_args),
                            ncols=100,
                        ):
                            del_H_i, H_i, H_true_i, iter_i = result
                            del_H[iter_i] = del_H_i
                            H[iter_i] = H_i
                            H_true[iter_i] = H_true_i
                else:
                    # with map, without tqdm
                    for result in pool.map(cross_entropy_difference, input_args):
                        del_H_i, H_i, H_true_i, iter_i = result
                        del_H[iter_i] = del_H_i
                        H[iter_i] = H_i
                        H_true[iter_i] = H_true_i

        best_thr = self.find_best_SNR_threshold(snr_thrs, del_H)
        if print_output:
            print(f"Best SNR threshold: {best_thr:.2f}")

        return best_thr, del_H, H, H_true, snr_thrs

    def find_best_SNR_threshold(self, thrs, del_H):
        """
        Function to find the best SNR threshold using spline interpolation and optimization.
        
        Parameters
        ----------
        thrs : np.ndarray
            Array of SNR thresholds tested.
        del_H : np.ndarray
            Array of cross-entropy differences for each threshold tested.

        Returns
        -------
        best_thr : float
            The optimal SNR threshold that maximizes the cross-entropy difference.
        """
        
        spline = interp1d(thrs, del_H, kind='cubic')
        min_bound = np.min(thrs)
        max_bound = np.max(thrs)
        best_thr = minimize_scalar(lambda x: -spline(x), bounds=(min_bound, max_bound), method='bounded').x

        return best_thr
    
    def _multiprocessing_error(self):
        """
        Prints an error message when multiprocessing is used.
        """
        # to access multi-cores instead of multithreading
        if mpp.current_process().name != 'MainProcess':
            print(
                "\n\n[ERROR] This multiprocessing code must be run under 'if __name__ == \"__main__\":'.\n"
                "Please wrap your script entry point in this guard.\n"
                "See: https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming\n"
            )
            raise RuntimeError(
                "\nMultiprocessing code must be run under 'if __name__ == \"__main__\":'.\n\n"
            )