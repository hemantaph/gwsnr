:py:mod:`gwsnr.utils`
=====================

.. py:module:: gwsnr.utils

.. autoapi-nested-parse::

   Helper functions for gwsnr

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.utils.NumpyEncoder



Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.utils.dealing_with_psds
   gwsnr.utils.power_spectral_density_pycbc
   gwsnr.utils.interpolator_check
   gwsnr.utils.interpolator_pickle_path
   gwsnr.utils.load_json
   gwsnr.utils.save_json
   gwsnr.utils.save_json_dict
   gwsnr.utils.load_json_dict



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

   
   Function for interpolator (snr_halfsacaled) check and generation if not exists.


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

   
   Function for storing or getting interpolator (snr_halfsacaled) pickle path


   :Parameters:

       **param_dict_given** : dict
           dictionary of parameters based on which the existence of interpolator will be checked

       **path** : str
           path to the directory where the interpolator pickle file will be stored

   :Returns:

       **path_interpolator** : str
           path to the interpolator pickle file
           e.g. './interpolator_pickle/L1/halfSNR_dict_0.pickle'

       it_exist: bool
           True if the interpolator exists
           False if the interpolator does not exists













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

.. py:function:: save_json(param, file_name)

   
   Save a json file.


   :Parameters:

       **param** : `dict`
           dictionary of parameters.

       **file_name** : `str`
           json file name for storing the parameters.














   ..
       !! processed by numpydoc !!

.. py:function:: save_json_dict(dict, file_name)

   
   Save a json file.


   :Parameters:

       **param** : `dict`
           dictionary of parameters.

       **file_name** : `str`
           json file name for storing the parameters.














   ..
       !! processed by numpydoc !!

.. py:function:: load_json_dict(file_name)

   
   Load a json file.


   :Parameters:

       **file_name** : `str`
           json file name for storing the parameters.

   :Returns:

       **param** : `dict`
           ..













   ..
       !! processed by numpydoc !!

