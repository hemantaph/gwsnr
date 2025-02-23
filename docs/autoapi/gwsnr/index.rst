:py:mod:`gwsnr`
===============

.. py:module:: gwsnr

.. autoapi-nested-parse::

   
   GWSNR: Gravitational Wave Signal-to-Noise Ratio
















   ..
       !! processed by numpydoc !!


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   gwsnr/index.rst
   multiprocessing_routine/index.rst
   njit_functions/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.GWSNR
   gwsnr.NumpyEncoder



Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.dealing_with_psds
   gwsnr.interpolator_check
   gwsnr.load_json
   gwsnr.load_pickle
   gwsnr.save_pickle
   gwsnr.save_json
   gwsnr.load_ann_h5_from_module
   gwsnr.load_ann_h5
   gwsnr.load_pickle_from_module
   gwsnr.load_json_from_module
   gwsnr.get_interpolated_snr
   gwsnr.findchirp_chirptime
   gwsnr.antenna_response
   gwsnr.antenna_response_array
   gwsnr.noise_weighted_inner_prod
   gwsnr.findchirp_chirptime
   gwsnr.einsum1
   gwsnr.einsum2
   gwsnr.gps_to_gmst
   gwsnr.ra_dec_to_theta_phi
   gwsnr.get_polarization_tensor
   gwsnr.antenna_response
   gwsnr.antenna_response_array
   gwsnr.noise_weighted_inner_product
   gwsnr.get_interpolated_snr
   gwsnr.cubic_spline_interpolator2d
   gwsnr.cubic_spline_interpolator
   gwsnr.coefficients_generator
   gwsnr.noise_weighted_inner_product
   gwsnr.noise_weighted_inner_prod
   gwsnr.save_json
   gwsnr.load_json
   gwsnr.save_pickle
   gwsnr.load_pickle
   gwsnr.load_ann_h5
   gwsnr.load_ann_h5_from_module
   gwsnr.load_json_from_module
   gwsnr.load_pickle_from_module
   gwsnr.dealing_with_psds
   gwsnr.power_spectral_density_pycbc
   gwsnr.interpolator_check
   gwsnr.interpolator_pickle_path



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

.. py:function:: save_pickle(file_name, param)

   
   Save a dictionary as a pickle file.


   :Parameters:

       **file_name** : `str`
           pickle file name for storing the parameters.

       **param** : `dict`
           dictionary to be saved as a pickle file.














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

.. py:function:: get_interpolated_snr(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr)

   
   Function to calculate the interpolated snr for a given set of parameters


   :Parameters:

       **mass_1** : `numpy.ndarray`
           Mass of the first body in solar masses.

       **mass_2** : `numpy.ndarray`
           Mass of the second body in solar masses.

       **luminosity_distance** : `float`
           Luminosity distance to the source in Mpc.

       **theta_jn** : `numpy.ndarray`
           Angle between the total angular momentum and the line of sight to the source in radians.

       **psi** : `numpy.ndarray`
           Polarization angle of the source.

       **geocent_time** : `numpy.ndarray`
           GPS time of the source.

       **ra** : ``numpy.ndarray`
           Right ascension of the source in radians.

       **dec** : `numpy.ndarray`
           Declination of the source in radians.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix)

       **snr_partialscaled** : `numpy.ndarray`
           Array of snr_partialscaled coefficients for the detector.

       **ratio_arr** : `numpy.ndarray`
           Array of mass ratio values for the snr_partialscaled coefficients.

       **mtot_arr** : `numpy.ndarray`
           Array of total mass values for the snr_partialscaled coefficients.

   :Returns:

       **snr** : `float`
           snr of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: findchirp_chirptime(m1, m2, fmin)

   
   Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.


   :Parameters:

       **m1** : `float`
           Mass of the first body in solar masses.

       **m2** : `float`
           Mass of the second body in solar masses.

       **fmin** : `float`
           Lower frequency cutoff.

   :Returns:

       **chirp_time** : float
           Time taken from f_min to f_lso (last stable orbit frequency).













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response(ra, dec, time, psi, detector_tensor, mode='plus')

   
   Function to calculate the antenna response


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : float
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix)

       **mode** : `str`
           Mode of the polarization. Default is 'plus'.

   :Returns:

       antenna_response: `float`
           Antenna response of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_array(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the antenna response in array form.


   :Parameters:

       **ra** : `numpy.ndarray`
           Right ascension of the source in radians.

       **dec** : `numpy.ndarray`
           Declination of the source in radians.

       **time** : `numpy.ndarray`
           GPS time of the source.

       **psi** : `numpy.ndarray`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the multiple detectors (nx3x3 matrix), where n is the number of detectors.

   :Returns:

       antenna_response: `numpy.ndarray`
           Antenna response of the detector. Shape is (n, len(ra)).













   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_prod(params)

   
   Probaility of detection of GW for the given sensitivity of the detectors


   :Parameters:

       **params** : list
           list of parameters for the inner product calculation
           List contains:

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

   :Returns:

       **SNRs_list** : list
           contains opt_snr for each detector and net_opt_snr

       **params[19]** : int
           index tracker













   ..
       !! processed by numpydoc !!

.. py:class:: GWSNR(npool=int(4), mtot_min=2.0, mtot_max=439.6, ratio_min=0.1, ratio_max=1.0, mtot_resolution=500, ratio_resolution=50, sampling_frequency=2048.0, waveform_approximant='IMRPhenomD', minimum_frequency=20.0, duration_max=None, snr_type='interpolation', psds=None, ifos=None, interpolator_dir='./interpolator_pickle', create_new_interpolator=False, gwsnr_verbose=True, multiprocessing_verbose=True, mtot_cut=True, pdet=False, snr_th=8.0, snr_th_net=8.0, ann_path_dict=None)


   
   Class to calculate SNR of a CBC signal with either interpolation or inner product method. Interpolation method is much faster than inner product method. Interpolation method is tested for IMRPhenomD and TaylorF2 waveform approximants for the spinless scenario.


   :Parameters:

       **npool** : `int`
           Number of processors to use for parallel processing.
           Default is 4.

       **mtot_min** : `float`
           Minimum total mass of the binary in solar mass (use interpolation purpose). Default is 2.0.

       **mtot_max** : `float`
           Maximum total mass of the binary in solar mass (use interpolation purpose). Default is 184. This is set so that the waveform is within the frequency range of the detector (with fmin=20.).

       **ratio_min** : `float`
           Minimum mass ratio of the binary (use interpolation purpose). Default is 0.1.

       **ratio_max** : `float`
           Maximum mass ratio of the binary (use interpolation purpose). Default is 1.0.

       **mtot_resolution** : `int`
           Number of points in the total mass array (use interpolation purpose). Default is 100.

       **ratio_resolution** : `int`
           Number of points in the mass ratio array (use interpolation purpose). Default is 100.

       **sampling_frequency** : `float`
           Sampling frequency of the detector. Default is 2048.0.

       **waveform_approximant** : `str`
           Waveform approximant to use. Default is 'IMRPhenomD'.

       **minimum_frequency** : `float`
           Minimum frequency of the waveform. Default is 20.0.

       **snr_type** : `str`
           Type of SNR calculation. Default is 'interpolation'.
           options: 'interpolation', 'inner_product', 'pdet', 'ann'

       **psds** : `dict`
           Dictionary of psds for different detectors. Default is None. If None, bilby's default psds will be used. Other options:

           Example 1: when values are psd name from pycbc analytical psds, psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044','V1':'AdvVirgo'}. To check available psd name run

           >>> import pycbc.psd
           >>> pycbc.psd.get_lalsim_psd_list()
           Example 2: when values are psd txt file available in bilby,
           psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt', 'K1':'KAGRA_design_asd.txt'}.
           For other psd files, check https://github.com/lscsoft/bilby/tree/master/bilby/gw/detector/noise_curves

           Example 3: when values are custom psd txt file. psds={'L1':'custom_psd.txt','H1':'custom_psd.txt'}. Custom created txt file has two columns. 1st column: frequency array, 2nd column: strain.

       **ifos** : `list` or `None`
           List of interferometer objects or detector names. Default is None. If None, bilby's default interferometer objects will be used. For example for LIGO India detector, it can be defined as follows,

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
           >>> snr = GWSNR(psds=dict(LIO='your_asd.txt'), ifos=[ifosLIO])

       **interpolator_dir** : `str`
           Path to store the interpolator pickle file. Default is './interpolator_pickle'.

       **create_new_interpolator** : `bool`
           If set True, new interpolator will be generated or replace the existing one. Default is False.

       **gwsnr_verbose** : `bool`
           If True, print all the parameters of the class instance. Default is True.

       **multiprocessing_verbose** : `bool`
           If True, it will show progress bar while computing SNR (inner product) with :meth:`~snr_with_interpolation`. Default is True. If False, it will not show progress bar but will be faster.

       **mtot_cut** : `bool`
           If True, it will set the maximum total mass of the binary according to the minimum frequency of the waveform. Default is True.











   .. rubric:: Examples

   >>> from gwsnr import GWSNR
   >>> snr = GWSNR()
   >>> snr.snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)

   Instance Attributes
   ----------
   GWSNR class has the following attributes,

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
   |:attr:`~interpolator_dir`            | `str`                            |
   +-------------------------------------+----------------------------------+
   |:attr:`~psds_list`                   | `list` of bilby's                |
   |                                     |  PowerSpectralDensity `object`   |
   +-------------------------------------+----------------------------------+
   |:attr:`~detector_tensor_list`        | `list` of detector tensor        |
   |                                     |  `numpy.ndarray`                 |
   +-------------------------------------+----------------------------------+
   |:attr:`~detector_list`               | `list` of `str`                  |
   +-------------------------------------+----------------------------------+
   |:attr:`~path_interpolator`           | `list` of `str`                  |
   +-------------------------------------+----------------------------------+
   |:attr:`~multiprocessing_verbose`     | `bool`                           |
   +-------------------------------------+----------------------------------+

   Instance Methods
   ----------
   GWSNR class has the following methods,

   +-------------------------------------+----------------------------------+
   | Methods                             | Description                      |
   +=====================================+==================================+
   |:meth:`~snr`                         | Calls                            |
   |                                     | :meth:`~snr_with_interpolation`  |
   |                                     | or :meth:`~compute_bilby_snr`    |
   |                                     | depending on the value of        |
   |                                     | :attr:`~snr_type` attribute.     |
   +-------------------------------------+----------------------------------+
   |:meth:`~snr_with_interpolation`      | Calculates SNR using             |
   |                                     | interpolation method.            |
   +-------------------------------------+----------------------------------+
   |:meth:`~compute_bilby_snr`           | Calculates SNR using             |
   |                                     | inner product method.            |
   +-------------------------------------+----------------------------------+
   |:meth:`~bns_horizon`                 | Calculates BNS horizon           |
   |                                     | distance.                        |
   +-------------------------------------+----------------------------------+
   |:meth:`~print_all_params`            | Prints all the parameters of     |
   |                                     | the class instance.              |
   +-------------------------------------+----------------------------------+
   |:meth:`~init_partialscaled`             | Generates partialscaled SNR         |
   |                                     | interpolation coefficients.      |
   +-------------------------------------+----------------------------------+



   ..
       !! processed by numpydoc !!
   .. py:attribute:: npool

      
      ``int``

      Number of processors to use for parallel processing.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_min

      
      ``float``

      Minimum total mass of the binary in solar mass (use interpolation purpose).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_max

      
      ``float``

      Maximum total mass of the binary in solar mass (use interpolation purpose).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_min

      
      ``float``

      Minimum mass ratio of the binary (use interpolation purpose).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_max

      
      ``float``

      Maximum mass ratio of the binary (use interpolation purpose).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_resolution

      
      ``int``

      Number of points in the total mass array (use interpolation purpose).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_resolution

      
      ``int``

      Number of points in the mass ratio array (use interpolation purpose).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_arr

      
      ``numpy.ndarray``

      Array of mass ratio.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_partialsacaled

      
      ``numpy.ndarray``

      Array of partial scaled SNR interpolation coefficients.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: sampling_frequency

      
      ``float``

      Sampling frequency of the detector.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: waveform_approximant

      
      ``str``

      Waveform approximant to use.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_min

      
      ``float``

      Minimum frequency of the waveform.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_type

      
      ``str``

      Type of SNR calculation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: psds

      
      ``dict``

      Dictionary of psds for different detectors.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: interpolator_dir

      
      ``str``

      Path to store the interpolator pickle file.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: detector_list

      
      ``list``

      List of detectors.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: stored_snrs

      
      ``dict``

      Dictionary of stored SNRs.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: pdet

      
      ``bool``

      If True, it will calculate the probability of detection. Default is False. Can also be 'matched_filter' or 'bool'. The value 'True' and 'bool' will give the same result.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_th

      
      ``float``

      SNR threshold for individual detector. Use for pdet calculation. Default is 8.0.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_th_net

      
      ``float``

      SNR threshold for network SNR. Use for pdet calculation. Default is 8.0.















      ..
          !! processed by numpydoc !!

   .. py:method:: interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

      
      Function to generate the partialscaled SNR interpolator and return its pickle file paths.


      :Parameters:

          **interpolator_dir** : `str`
              Path to store the interpolator pickle file.

          **create_new_interpolator** : `bool`
              If set True, new interpolator will be generated or replace the existing one.

          **psds_list** : `list`
              List of psds for different detectors.

          **detector_tensor_list** : `list`
              List of detector tensor.

          **detector_list** : `list`
              List of detectors.

      :Returns:

          **path_interpolator_all** : `list`
              List of partialscaled SNR interpolator pickle file paths.













      ..
          !! processed by numpydoc !!

   .. py:method:: ann_initilization(ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant, snr_th)

      
      Function to initialize ANN model and scaler for the given detector list. It also generates the partialscaledSNR interpolator for the required waveform approximant.
















      ..
          !! processed by numpydoc !!

   .. py:method:: calculate_mtot_max(mtot_max, minimum_frequency)

      
      Function to calculate maximum total mass of the binary in solar mass (use in interpolation purpose) according to the minimum frequency of the waveform.


      :Parameters:

          **mtot_max** : `float`
              Maximum total mass of the binary in solar mass (use interpolation purpose).

          **minimum_frequency** : `float`
              Minimum frequency of the waveform.

      :Returns:

          **mtot_max** : `float`
              Maximum total mass of the binary in solar mass (use interpolation purpose).













      ..
          !! processed by numpydoc !!

   .. py:method:: print_all_params(verbose=True)

      
      Function to print all the parameters of the class instance


      :Parameters:

          **verbose** : `bool`
              If True, print all the parameters of the class instance. Default is True.














      ..
          !! processed by numpydoc !!

   .. py:method:: snr(mass_1=np.array([10.0]), mass_2=np.array([10.0]), luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Function for calling SNR calculation function depending on the value of snr_type attribute. If snr_type is 'interpolation', it calls snr_with_interpolation function. If snr_type is 'inner_product', it calls compute_bilby_snr function.


      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar mass. Default is 10.0.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar mass. Default is 10.0.

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle of the binary in radian. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Polarization angle of the binary in radian. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Phase of the binary in radian. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              Geocentric time of the binary in gps. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the binary in radian. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the binary in radian. Default is 0.0.

          **a_1** : `numpy.ndarray` or `float`
              Primary spin of the binary. Default is 0.0.

          **a_2** : `numpy.ndarray` or `float`
              Secondary spin of the binary. Default is 0.0.

          **tilt_1** : `numpy.ndarray` or `float`
              Tilt of the primary spin of the binary. Default is 0.0.

          **tilt_2** : `numpy.ndarray` or `float`
              Tilt of the secondary spin of the binary. Default is 0.0.

          **phi_12** : `numpy.ndarray` or `float`
              Relative angle between the primary and secondary spin of the binary. Default is 0.0.

          **phi_jl** : `numpy.ndarray` or `float`
              Angle between the total angular momentum and the orbital angular momentum of the binary. Default is 0.0.

          **gw_param_dict** : `dict`
              This allows to pass all the parameters as a dictionary (dict.keys()=param_names, dict.values()=param values). Default is False.

          **output_jsonfile** : `str` or `bool`
              If str, the SNR dictionary will be saved as a json file with the given name. Default is False.

      :Returns:

          **snr_dict** : `dict`
              Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).










      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> snr = GWSNR(snrs_type='interpolation')
      >>> snr.snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)



      ..
          !! processed by numpydoc !!

   .. py:method:: snr_with_ann(mass_1, mass_2, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, output_jsonfile=False)

      
      Function to calculate SNR using bicubic interpolation method.


      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar mass. Default is 10.0.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar mass. Default is 10.0.

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle of the binary in radian. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Polarization angle of the binary in radian. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Phase of the binary in radian. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              Geocentric time of the binary in gps. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the binary in radian. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the binary in radian. Default is 0.0.

          **output_jsonfile** : `str` or `bool`
              If str, the SNR dictionary will be saved as a json file with the given name. Default is False.

      :Returns:

          **snr_dict** : `dict`
              Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).










      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> snr = GWSNR(snr_type='ann', waveform_approximant='IMRPhenomXPHM')
      >>> snr.snr_with_ann(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)



      ..
          !! processed by numpydoc !!

   .. py:method:: output_ann(idx, params)

      
      Function to output the input data for ANN.


      :Parameters:

          **idx** : `numpy.ndarray`
              Index array.

          **params** : `dict`
              Dictionary of input parameters.

      :Returns:

          **X_L1** : `numpy.ndarray`
              Feature scaled input data for L1 detector.

          **X_H1** : `numpy.ndarray`
              Feature scaled input data for H1 detector.

          **X_V1** : `numpy.ndarray`
              Feature scaled input data for V1 detector.













      ..
          !! processed by numpydoc !!

   .. py:method:: snr_with_interpolation(mass_1, mass_2, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, output_jsonfile=False)

      
      Function to calculate SNR using bicubic interpolation method.


      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar mass. Default is 10.0.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar mass. Default is 10.0.

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle of the binary in radian. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Polarization angle of the binary in radian. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Phase of the binary in radian. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              Geocentric time of the binary in gps. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the binary in radian. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the binary in radian. Default is 0.0.

          **output_jsonfile** : `str` or `bool`
              If str, the SNR dictionary will be saved as a json file with the given name. Default is False.

      :Returns:

          **snr_dict** : `dict`
              Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).










      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> snr = GWSNR(snr_type='interpolation')
      >>> snr.snr_with_interpolation(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)



      ..
          !! processed by numpydoc !!

   .. py:method:: init_partialscaled()

      
      Function to generate partialscaled SNR interpolation coefficients. It will save the interpolator in the pickle file path indicated by the path_interpolator attribute.
















      ..
          !! processed by numpydoc !!

   .. py:method:: compute_bilby_snr(mass_1=10, mass_2=10, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, gw_param_dict=False, output_jsonfile=False)

      
      SNR calculated using inner product method. This is similar to the SNR calculation method used in bilby.


      :Parameters:

          **mass_1** : float
              The mass of the heavier object in the binary in solar masses.

          **mass_2** : float
              The mass of the lighter object in the binary in solar masses.

          **luminosity_distance** : float
              The luminosity distance to the binary in megaparsecs.

          **theta_jn** : float, optional
              The angle between the total angular momentum and the line of sight.
              Default is 0.

          **psi** : float, optional
              The gravitational wave polarisation angle.
              Default is 0.

          **phase** : float, optional
              The gravitational wave phase at coalescence.
              Default is 0.

          **geocent_time** : float, optional
              The GPS time of coalescence.
              Default is 1249852157.0.

          **ra** : float, optional
              The right ascension of the source.
              Default is 0.

          **dec** : float, optional
              The declination of the source.
              Default is 0.

          **a_1** : float, optional
              The spin magnitude of the heavier object in the binary.
              Default is 0.

          **tilt_1** : float, optional
              The tilt angle of the heavier object in the binary.
              Default is 0.

          **phi_12** : float, optional
              The azimuthal angle between the two spins.
              Default is 0.

          **a_2** : float, optional
              The spin magnitude of the lighter object in the binary.
              Default is 0.

          **tilt_2** : float, optional
              The tilt angle of the lighter object in the binary.
              Default is 0.

          **phi_jl** : float, optional
              The azimuthal angle between the total angular momentum and the orbital angular momentum.
              Default is 0.

          **verbose** : bool, optional
              If true, print the SNR.
              Default is True.

          **jsonFile** : bool, optional
              If true, save the SNR parameters and values in a json file.
              Default is False.

      :Returns:

          **snr_dict** : `dict`
              Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).










      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> snr = GWSNR(snrs_type='inner_product')
      >>> snr.compute_bilby_snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)



      ..
          !! processed by numpydoc !!

   .. py:method:: probability_of_detection(snr_dict, snr_th=None, snr_th_net=None, type='matched_filter')

      
      Probaility of detection of GW for the given sensitivity of the detectors


      :Parameters:

          **snr_dict** : `dict`
              Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).

          **rho_th** : `float`
              Threshold SNR for detection. Default is 8.0.

          **rho_net_th** : `float`
              Threshold net SNR for detection. Default is 8.0.

          **type** : `str`
              Type of SNR calculation. Default is 'matched_filter'. Other option is 'bool'.

      :Returns:

          **pdet_dict** : `dict`
              Dictionary of probability of detection for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=pdet_arrays).













      ..
          !! processed by numpydoc !!

   .. py:method:: detector_horizon(mass_1=1.4, mass_2=1.4, snr_th=None, snr_th_net=None)

      
      Function for finding detector horizon distance for BNS (m1=m2=1.4)


      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar mass. Default is 1.4.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar mass. Default is 1.4.

          **snr_th** : `float`
              SNR threshold for detection. Default is 8.0.

      :Returns:

          **horizon** : `dict`
              Dictionary of horizon distance for each detector in Mpc (dict.keys()=detector_names, dict.values()=horizon_distance).













      ..
          !! processed by numpydoc !!


.. py:function:: findchirp_chirptime(m1, m2, fmin)

   
   Time taken from f_min to f_lso (last stable orbit). 3.5PN in fourier phase considered.


   :Parameters:

       **m1** : `float`
           Mass of the first body in solar masses.

       **m2** : `float`
           Mass of the second body in solar masses.

       **fmin** : `float`
           Lower frequency cutoff.

   :Returns:

       **chirp_time** : float
           Time taken from f_min to f_lso (last stable orbit frequency).













   ..
       !! processed by numpydoc !!

.. py:function:: einsum1(m, n)

   
   Function to calculate einsum of two 3x1 vectors


   :Parameters:

       **m** : `numpy.ndarray`
           3x1 vector.

       **n** : `numpy.ndarray`
           3x1 vector.

   :Returns:

       **ans** : `numpy.ndarray`
           3x3 matrix.













   ..
       !! processed by numpydoc !!

.. py:function:: einsum2(m, n)

   
   Function to calculate einsum of two 3x3 matrices


   :Parameters:

       **m** : `numpy.ndarray`
           3x3 matrix.

       **n** : `numpy.ndarray`
           3x3 matrix.

   :Returns:

       **ans** : `numpy.ndarray`
           3x3 matrix.













   ..
       !! processed by numpydoc !!

.. py:function:: gps_to_gmst(gps_time)

   
   Function to convert gps time to greenwich mean sidereal time


   :Parameters:

       **gps_time** : `float`
           GPS time in seconds.

   :Returns:

       **gmst** : `float`
           Greenwich mean sidereal time in radians.













   ..
       !! processed by numpydoc !!

.. py:function:: ra_dec_to_theta_phi(ra, dec, gmst)

   
   Function to convert ra and dec to theta and phi


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : `float`
           Declination of the source in radians.

       **gmst** : `float`
           Greenwich mean sidereal time in radians.

   :Returns:

       **theta** : `float`
           Polar angle in radians.

       **phi** : `float`
           Azimuthal angle in radians.













   ..
       !! processed by numpydoc !!

.. py:function:: get_polarization_tensor(ra, dec, time, psi, mode='plus')

   
   Function to calculate the polarization tensor


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : float
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

       **mode** : `str`
           Mode of the polarization. Default is 'plus'.

   :Returns:

       polarization_tensor: `numpy.ndarray`
           Polarization tensor of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response(ra, dec, time, psi, detector_tensor, mode='plus')

   
   Function to calculate the antenna response


   :Parameters:

       **ra** : `float`
           Right ascension of the source in radians.

       **dec** : float
           Declination of the source in radians.

       **time** : `float`
           GPS time of the source.

       **psi** : `float`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix)

       **mode** : `str`
           Mode of the polarization. Default is 'plus'.

   :Returns:

       antenna_response: `float`
           Antenna response of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: antenna_response_array(ra, dec, time, psi, detector_tensor)

   
   Function to calculate the antenna response in array form.


   :Parameters:

       **ra** : `numpy.ndarray`
           Right ascension of the source in radians.

       **dec** : `numpy.ndarray`
           Declination of the source in radians.

       **time** : `numpy.ndarray`
           GPS time of the source.

       **psi** : `numpy.ndarray`
           Polarization angle of the source.

       **detector_tensor** : array-like
           Detector tensor for the multiple detectors (nx3x3 matrix), where n is the number of detectors.

   :Returns:

       antenna_response: `numpy.ndarray`
           Antenna response of the detector. Shape is (n, len(ra)).













   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_product(signal1, signal2, psd, duration)

   
   Noise weighted inner product of two time series data sets.


   :Parameters:

       **signal1: `numpy.ndarray` or `float`**
           First series data set.

       **signal2: `numpy.ndarray` or `float`**
           Second series data set.

       **psd: `numpy.ndarray` or `float`**
           Power spectral density of the detector.

       **duration: `float`**
           Duration of the data.














   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr)

   
   Function to calculate the interpolated snr for a given set of parameters


   :Parameters:

       **mass_1** : `numpy.ndarray`
           Mass of the first body in solar masses.

       **mass_2** : `numpy.ndarray`
           Mass of the second body in solar masses.

       **luminosity_distance** : `float`
           Luminosity distance to the source in Mpc.

       **theta_jn** : `numpy.ndarray`
           Angle between the total angular momentum and the line of sight to the source in radians.

       **psi** : `numpy.ndarray`
           Polarization angle of the source.

       **geocent_time** : `numpy.ndarray`
           GPS time of the source.

       **ra** : ``numpy.ndarray`
           Right ascension of the source in radians.

       **dec** : `numpy.ndarray`
           Declination of the source in radians.

       **detector_tensor** : array-like
           Detector tensor for the detector (3x3 matrix)

       **snr_partialscaled** : `numpy.ndarray`
           Array of snr_partialscaled coefficients for the detector.

       **ratio_arr** : `numpy.ndarray`
           Array of mass ratio values for the snr_partialscaled coefficients.

       **mtot_arr** : `numpy.ndarray`
           Array of total mass values for the snr_partialscaled coefficients.

   :Returns:

       **snr** : `float`
           snr of the detector.













   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_interpolator2d(xnew, ynew, coefficients, x, y)

   
   Function to calculate the interpolated value of snr_partialscaled given the mass ratio (ynew) and total mass (xnew). This is based off 2D bicubic spline interpolation.


   :Parameters:

       **xnew** : `float`
           Total mass of the binary.

       **ynew** : `float`
           Mass ratio of the binary.

       **coefficients** : `numpy.ndarray`
           Array of coefficients for the cubic spline interpolation.

       **x** : `numpy.ndarray`
           Array of total mass values for the coefficients.

       **y** : `numpy.ndarray`
           Array of mass ratio values for the coefficients.

   :Returns:

       **result** : `float`
           Interpolated value of snr_partialscaled.













   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_interpolator(xnew, coefficients, x)

   
   Function to calculate the interpolated value of snr_partialscaled given the total mass (xnew). This is based off 1D cubic spline interpolation.


   :Parameters:

       **xnew** : `float`
           Total mass of the binary.

       **coefficients** : `numpy.ndarray`
           Array of coefficients for the cubic spline interpolation.

       **x** : `numpy.ndarray`
           Array of total mass values for the coefficients.

   :Returns:

       **result** : `float`
           Interpolated value of snr_partialscaled.













   ..
       !! processed by numpydoc !!

.. py:function:: coefficients_generator(y1, y2, y3, y4, z1, z2, z3, z4)

   
   Function to generate the coefficients for the cubic spline interpolation of fn(y)=z.


   :Parameters:

       **y1, y2, y3, y4, z1, z2, z3, z4: `float`**
           Values of y and z for the cubic spline interpolation.

   :Returns:

       coefficients: `numpy.ndarray`
           Coefficients for the cubic spline interpolation.













   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_product(signal1, signal2, psd, duration)

   
   Noise weighted inner product of two time series data sets.


   :Parameters:

       **signal1: `numpy.ndarray` or `float`**
           First series data set.

       **signal2: `numpy.ndarray` or `float`**
           Second series data set.

       **psd: `numpy.ndarray` or `float`**
           Power spectral density of the detector.

       **duration: `float`**
           Duration of the data.














   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_prod(params)

   
   Probaility of detection of GW for the given sensitivity of the detectors


   :Parameters:

       **params** : list
           list of parameters for the inner product calculation
           List contains:

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

   :Returns:

       **SNRs_list** : list
           contains opt_snr for each detector and net_opt_snr

       **params[19]** : int
           index tracker













   ..
       !! processed by numpydoc !!

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

