:py:mod:`gwsnr.utils.utils`
===========================

.. py:module:: gwsnr.utils.utils

.. autoapi-nested-parse::

   Helper functions for gwsnr

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.utils.utils.NumpyEncoder



Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.utils.utils.save_json
   gwsnr.utils.utils.load_json
   gwsnr.utils.utils.save_pickle
   gwsnr.utils.utils.load_pickle
   gwsnr.utils.utils.load_ann_h5
   gwsnr.utils.utils.append_json
   gwsnr.utils.utils.add_dictionaries_together
   gwsnr.utils.utils.get_param_from_json
   gwsnr.utils.utils.load_ann_h5_from_module
   gwsnr.utils.utils.load_json_from_module
   gwsnr.utils.utils.load_pickle_from_module
   gwsnr.utils.utils.dealing_with_psds
   gwsnr.utils.utils.power_spectral_density_bilby
   gwsnr.utils.utils.power_spectral_density_pycbc
   gwsnr.utils.utils.interpolator_check
   gwsnr.utils.utils.interpolator_pickle_path



.. py:class:: NumpyEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)


   Bases: :py:obj:`json.JSONEncoder`

   
   Custom JSON encoder for numpy data types. It converts numpy.ndarray objects (and any nested-list composition
   that includes ndarray objects) into regular lists for JSON serialization. This is particularly useful when
   serializing data structures that include numpy arrays.
















   ..
       !! processed by numpydoc !!
   .. py:method:: default(obj)

      
      Implement this method in a subclass such that it returns
      a serializable object for ``o``, or calls the base implementation
      (to raise a ``TypeError``).

      For example, to support arbitrary iterators, you could
      implement default like this::

          def default(self, o):
              try:
                  iterable = iter(o)
              except TypeError:
                  pass
              else:
                  return list(iterable)
              # Let the base class default method raise the TypeError
              return JSONEncoder.default(self, o)















      ..
          !! processed by numpydoc !!


.. py:function:: save_json(file_name, param)

   
   Save a dictionary as a json file.


   :Parameters:

       **file_name** : `str`
           json file name for storing the parameters.

       **param** : `dict`
           dictionary to be saved as a json file.














   ..
       !! processed by numpydoc !!

.. py:function:: load_json(file_name)

   
   Load a json file.


   :Parameters:

       **file_name** : `str`
           json file name for storing the parameters.

   :Returns:

       **param** : `dict`
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: save_pickle(file_name, param)

   
   Save a dictionary as a pickle file.


   :Parameters:

       **file_name** : `str`
           pickle file name for storing the parameters.

       **param** : `dict`
           dictionary to be saved as a pickle file.














   ..
       !! processed by numpydoc !!

.. py:function:: load_pickle(file_name)

   
   Load a pickle file.


   :Parameters:

       **file_name** : `str`
           pickle file name for storing the parameters.

   :Returns:

       **param** : `dict`
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: load_ann_h5(filename)

   
   Function to load a specific dataset from an .h5 file


   :Parameters:

       **filename** : str
           name of the .h5 file

   :Returns:

       **model** : `keras.models.Model`
           Keras model loaded from the .h5 file













   ..
       !! processed by numpydoc !!

.. py:function:: append_json(file_name, new_dictionary, old_dictionary=None, replace=False)

   
   Append (values with corresponding keys) and update a json file with a dictionary. There are four options:

   1. If old_dictionary is provided, the values of the new dictionary will be appended to the old dictionary and save in the 'file_name' json file.
   2. If replace is True, replace the json file (with the 'file_name') content with the new_dictionary.
   3. If the file does not exist, create a new one with the new_dictionary.
   4. If none of the above, append the new dictionary to the content of the json file.

   :Parameters:

       **file_name** : `str`
           json file name for storing the parameters.

       **new_dictionary** : `dict`
           dictionary to be appended to the json file.

       **old_dictionary** : `dict`, optional
           If provided the values of the new dictionary will be appended to the old dictionary and save in the 'file_name' json file.
           Default is None.

       **replace** : `bool`, optional
           If True, replace the json file with the dictionary. Default is False.














   ..
       !! processed by numpydoc !!

.. py:function:: add_dictionaries_together(dictionary1, dictionary2)

   
   Adds two dictionaries with the same keys together.


   :Parameters:

       **dictionary1** : `dict`
           dictionary to be added.

       **dictionary2** : `dict`
           dictionary to be added.

   :Returns:

       **dictionary** : `dict`
           dictionary with added values.













   ..
       !! processed by numpydoc !!

.. py:function:: get_param_from_json(json_file)

   
   Function to get the parameters from json file.


   :Parameters:

       **json_file** : `str`
           json file name for storing the parameters.

   :Returns:

       **param** : `dict`
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: load_ann_h5_from_module(package, directory, filename)

   
   Function to load a specific dataset from an .h5 file within the package


   :Parameters:

       **package** : str
           name of the package

       **directory** : str
           name of the directory within the package

       **filename** : str
           name of the .h5 file

   :Returns:

       **model** : `keras.models.Model`
           Keras model loaded from the .h5 file













   ..
       !! processed by numpydoc !!

.. py:function:: load_json_from_module(package, directory, filename)

   
   Function to load a specific dataset from a .json file within the package


   :Parameters:

       **package** : str
           name of the package

       **directory** : str
           name of the directory within the package

       **filename** : str
           name of the .json file

   :Returns:

       **data** : `dict`
           Dictionary loaded from the .json file













   ..
       !! processed by numpydoc !!

.. py:function:: load_pickle_from_module(package, directory, filename)

   
   Function to load a specific dataset from a .pkl file within the package


   :Parameters:

       **package** : str
           name of the package

       **directory** : str
           name of the directory within the package

       **filename** : str
           name of the .pkl file

   :Returns:

       **data** : `dict`
           Dictionary loaded from the .pkl file













   ..
       !! processed by numpydoc !!

.. py:function:: dealing_with_psds(psds=None, ifos=None, f_min=20.0, sampling_frequency=2048.0)

   
   Function to deal with psds inputs and for creating bilby.gw.detector.PowerSpectralDensity objects.


   :Parameters:

       **psds** : dict
           dictionary of psds. psds.keys()=detector names, psds.values()=psds file names or pycbc psd names

       **ifos** : `list` or `None`
           List of interferometer objects or interferometer name list. Default is None. If None, bilby's default interferometer objects will be used.

       **f_min** : `float`
           Minimum frequency of the psds. Default is 20.

       **sampling_frequency** : `float`
           Sampling frequency of the psds. Default is 2048.

   :Returns:

       **psds_list** : `list`
           list of bilby.gw.detector.PowerSpectralDensity objects

       **detector_tensor_list** : `list`
           list of detector tensors

       **detector_list** : `list`
           list of detector names













   ..
       !! processed by numpydoc !!

.. py:function:: power_spectral_density_bilby(psd_txt)

   
   psd array finder from bilby


   :Parameters:

       **psd_txt** : str
           name of the psd
           e.g. 'aLIGOaLIGODesignSensitivityT1800044'

   :Returns:

       **psd_array** : bilby.gw.detector.psd.PowerSpectralDensity object
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: power_spectral_density_pycbc(psd, f_min=20.0, sampling_frequency=2048.0)

   
   psd array finder from pycbc


   :Parameters:

       **psd** : str
           name of the psd
           e.g. 'aLIGOaLIGODesignSensitivityT1800044'

       **f_min** : float
           minimum frequency of the psd
           default: 20.

       **sampling_frequency** : float
           sampling frequency of the psd
           default: 2048.

   :Returns:

       **psd_array** : bilby.gw.detector.psd.PowerSpectralDensity object
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: interpolator_check(param_dict_given, interpolator_dir, create_new)

   
   Function for interpolator (snr_partialsacaled) check and generation if not exists.


   :Parameters:

       **param_dict_given** : dict
           dictionary of parameters based on which the existence of interpolator will be checked

       **interpolator_dir** : str
           path to the interpolator pickle file

       **create_new** : bool
           if True, new interpolator will be generated even if the interpolator exists
           if False, existing interpolator will be used if exists, otherwise new interpolator will be generated

   :Returns:

       **psds_list_** : list
           list of psd objects

       **detector_tensor_list_** : list
           list of detector tensors

       **detector_list_** : list
           list of detector names













   ..
       !! processed by numpydoc !!

.. py:function:: interpolator_pickle_path(param_dict_given, path='./interpolator_pickle')

   
   Function for storing or getting interpolator (snr_partialsacaled) pickle path


   :Parameters:

       **param_dict_given** : dict
           dictionary of parameters based on which the existence of interpolator will be checked

       **path** : str
           path to the directory where the interpolator pickle file will be stored

   :Returns:

       **path_interpolator** : str
           path to the interpolator pickle file
           e.g. './interpolator_pickle/L1/partialSNR_dict_0.pickle'

       it_exist: bool
           True if the interpolator exists
           False if the interpolator does not exists













   ..
       !! processed by numpydoc !!

