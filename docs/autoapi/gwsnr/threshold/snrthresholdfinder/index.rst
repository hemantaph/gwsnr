:py:mod:`gwsnr.threshold.snrthresholdfinder`
============================================

.. py:module:: gwsnr.threshold.snrthresholdfinder

.. autoapi-nested-parse::

   This module implements the SNRThresholdFinder class to determine the optimal SNR threshold for gravitational wave detection using cross-entropy maximization (following Essick et al. 2023).

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.threshold.snrthresholdfinder.SNRThresholdFinder




.. py:class:: SNRThresholdFinder(catalog_file=None, npool=4, selection_range=None, original_detection_statistic=None, projected_detection_statistic=None, parameters_to_fit=None, sample_size=20000, multiprocessing_verbose=True)


   
   A class to find the optimal SNR threshold for gravitational wave detection using cross-entropy maximization.


   :Parameters:

       **catalog_file** : str
           Path to the HDF5 file containing the injection catalog data. The file should have something like the following structure (refer to https://zenodo.org/records/16740117):
           ```
           injections.hdf
           |-- events
           |   |-- z  (parameter to me fitted on)
           |   |-- mass1_source (parameter with which the data is to be selected with)
           |   |-- gstlal_far (original_detection_statistic)
           |   |-- observed_snr_net (projected_detection_statistic)
           ```

       **original_detection_statistic** : dict, optional
           Dictionary specifying the original detection statistic with keys:
           'parameter' (str): Name of the key in the catalog for the original detection statistic.
           'threshold' (float): Threshold value for the original detection statistic.
           Default is {'parameter': 'gstlal_far', 'threshold': 1}.

       **projected_detection_statistic** : dict, optional
           Dictionary specifying the projected detection statistic with keys:
           'parameter' (str): Name of the key in the catalog for the projected detection statistic.
           'threshold' (float): Threshold value for the projected detection statistic.
           'threshold_search_bounds' (tuple): Bounds for the threshold search.
           Default is {'parameter': 'observed_snr_net', 'threshold': None, 'threshold_search_bounds': (4, 14)}.

       **parameters_to_fit** : list of str, optional
           List of parameter to fit, e.g., ['redshift']. Default is ['redshift'].

       **sample_size** : int, optional
           Number of samples to use for KDE estimation. Default is 10000.

       **selection_range** : dict, optional
           Dictionary specifying the selection range with keys:
           'parameter' (str or list): Parameter(s) to apply the selection range on.
           'range' (tuple): Tuple specifying the (min, max) range for selection.
           Default is {'parameter': 'mass1_source', 'range': (5, 200)}.











   .. rubric:: Examples

   >>> finder = SNRThresholdFinder(catalog_file='injection_catalog.h5')
   >>> best_thr, del_H, H, H_true, snr_thrs = finder.find_threshold(iteration=10)
   >>> print(f"Best SNR threshold: {best_thr:.2f}")



   ..
       !! processed by numpydoc !!
   .. py:method:: det_data(catalog_file)

      
      Function to load and preprocess the injection catalog data from an HDF5 file.


      :Parameters:

          **catalog_file** : str
              Path to the HDF5 file containing the injection catalog data.

          **Returns**
              ..

          **-------**
              ..

          **result_dict** : dict
              Dictionary containing the preprocessed data for the specified parameters and detection statistics.





      :Raises:

          ValueError
              If 'redshift' is not included in parameters_to_fit.









      ..
          !! processed by numpydoc !!

   .. py:method:: find_threshold(iteration=10, print_output=True, no_multiprocessing=False)

      
      Function to find the optimal SNR threshold by maximizing the cross-entropy difference.


      :Parameters:

          **iteration** : int, optional
              Number of iterations for threshold search. Default is 10.

          **print_output** : bool, optional
              Whether to print the best SNR threshold. Default is True.

      :Returns:

          **best_thr** : float
              The optimal SNR threshold that maximizes the cross-entropy difference.

          **del_H** : np.ndarray
              Array of cross-entropy differences for each threshold tested.

          **H** : np.ndarray
              Array of cross-entropy values for the KDE with cut.

          **H_true** : np.ndarray
              Array of cross-entropy values for the original KDE.

          **snr_thrs** : np.ndarray
              Array of SNR thresholds tested.




      :Raises:

          ValueError
              If the number of iterations is less than 1.









      ..
          !! processed by numpydoc !!

   .. py:method:: find_best_SNR_threshold(thrs, del_H)

      
      Function to find the best SNR threshold using spline interpolation and optimization.


      :Parameters:

          **thrs** : np.ndarray
              Array of SNR thresholds tested.

          **del_H** : np.ndarray
              Array of cross-entropy differences for each threshold tested.

      :Returns:

          **best_thr** : float
              The optimal SNR threshold that maximizes the cross-entropy difference.













      ..
          !! processed by numpydoc !!


