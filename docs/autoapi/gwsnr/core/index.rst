:orphan:

:py:mod:`gwsnr.core`
====================

.. py:module:: gwsnr.core


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   gwsnr/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.core.GWSNR




.. py:class:: GWSNR(npool=int(4), mtot_min=2 * 4.98, mtot_max=2 * 112.5 + 10.0, ratio_min=0.1, ratio_max=1.0, spin_max=0.99, mtot_resolution=200, ratio_resolution=20, spin_resolution=10, batch_size_interpolation=1000000, sampling_frequency=2048.0, waveform_approximant='IMRPhenomD', frequency_domain_source_model='lal_binary_black_hole', minimum_frequency=20.0, reference_frequency=None, duration_max=None, duration_min=None, fixed_duration=None, snr_method='interpolation_no_spins', snr_type='optimal_snr', noise_realization=None, psds=None, ifos=None, interpolator_dir='./interpolator_pickle', create_new_interpolator=False, gwsnr_verbose=True, multiprocessing_verbose=True, mtot_cut=False, pdet=False, snr_th=8.0, snr_th_net=8.0, ann_path_dict=None, snr_recalculation=False, snr_recalculation_range=[4, 12], snr_recalculation_waveform_approximant='IMRPhenomXPHM')


   
   Class to calculate SNR of a CBC signal with either interpolation or inner product method. Interpolation method is much faster than inner product method. Interpolation method is tested for IMRPhenomD, TaylorF2, and IMRPhenomXPHM waveform approximants for both spinless and aligned-spin scenarios.


   :Parameters:

       **npool** : `int`
           Number of processors to use for parallel processing.
           Default is 4.

       **mtot_min** : `float`
           Minimum total mass of the binary in solar mass (use interpolation purpose). Default is 2*4.98-2 (4.98 Mo is the minimum component mass of BBH systems in GWTC-3).

       **mtot_max** : `float`
           Maximum total mass of the binary in solar mass (use interpolation purpose). Default is 2*112.5+2 (112.5 Mo is the maximum component mass of BBH systems in GWTC-3).
           This is automatically adjusted based on minimum_frequency if mtot_cut=True.

       **ratio_min** : `float`
           Minimum mass ratio of the binary (use interpolation purpose). Default is 0.1.

       **ratio_max** : `float`
           Maximum mass ratio of the binary (use interpolation purpose). Default is 1.0.

       **spin_max** : `float`
           Maximum spin magnitude for aligned-spin interpolation methods. Default is 0.9.

       **mtot_resolution** : `int`
           Number of points in the total mass array (use interpolation purpose). Default is 200.

       **ratio_resolution** : `int`
           Number of points in the mass ratio array (use interpolation purpose). Default is 50.

       **spin_resolution** : `int`
           Number of points in the spin arrays for aligned-spin interpolation methods. Default is 20.

       **sampling_frequency** : `float`
           Sampling frequency of the detector. Default is 2048.0.

       **waveform_approximant** : `str`
           Waveform approximant to use. Default is 'IMRPhenomD'.

       **frequency_domain_source_model** : `str`
           Source model for frequency domain waveform generation. Default is 'lal_binary_black_hole'.

       **minimum_frequency** : `float`
           Minimum frequency of the waveform. Default is 20.0.

       **reference_frequency** : `float` or `None`
           Reference frequency of the waveform. Default is None (sets to minimum_frequency).

       **duration_max** : `float` or `None`
           Maximum duration for waveform generation. Default is None. Automatically set to 64.0 for IMRPhenomXPHM on Intel processors.

       **duration_min** : `float` or `None`
           Minimum duration for waveform generation. Default is None.

       **snr_method** : `str`
           Type of SNR calculation. Default is 'interpolation'.
           options: 'interpolation', 'interpolation_no_spins', 'interpolation_no_spins_jax', 'interpolation_no_spins_mlx', 'interpolation_aligned_spins', 'interpolation_aligned_spins_jax', 'interpolation_aligned_spins_mlx', 'inner_product', 'inner_product_jax', 'ann'

       **psds** : `dict`
           Dictionary of psds for different detectors. Default is None. If None, bilby's default psds will be used. Other options:

           Example 1: when values are psd name from pycbc analytical psds, psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044','V1':'AdvVirgo'}. To check available psd name run

           >>> import pycbc.psd
           >>> pycbc.psd.get_lalsim_psd_list()
           Example 2: when values are psd txt file available in bilby,
           psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt'}.
           For other psd files, check https://github.com/lscsoft/bilby/tree/master/bilby/gw/detector/noise_curves

           Example 3: when values are custom psd txt file. psds={'L1':'custom_psd.txt','H1':'custom_psd.txt'}. Custom created txt file has two columns. 1st column: frequency array, 2nd column: strain.
           Example 4: when you want psds to be created from a stretch of data for a given trigger time. psds={'L1':1246527224.169434}

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
                   longitude = 77 + 1. / 60 + 51.0997 / 3600,
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
           If True, it will show progress bar while computing SNR (inner product) with :meth:`~optimal_snr_with_interpolation`. Default is True. If False, it will not show progress bar but will be faster.

       **mtot_cut** : `bool`
           If True, it will set the maximum total mass of the binary according to the minimum frequency of the waveform. This is done searching for the maximum total mass corresponding to zero chirp time, i.e. the sytem merge below the minimum frequency. This is done to avoid unnecessary computation of SNR for systems that will not be detected. Default is False.

       **pdet** : `bool` or `str`
           If True or 'bool', calculate probability of detection using boolean method. If 'matched_filter', use matched filter probability. Default is False.

       **snr_th** : `float`
           SNR threshold for individual detector for pdet calculation. Default is 8.0.

       **snr_th_net** : `float`
           SNR threshold for network SNR for pdet calculation. Default is 8.0.

       **ann_path_dict** : `dict` or `str` or `None`
           Dictionary or path to json file containing ANN model and scaler paths for different detectors. Default is None (uses built-in models).

       **snr_recalculation** : `bool`
           If True, enables hybrid SNR recalculation for systems near detection threshold. Default is False.

       **snr_recalculation_range** : `list`
           SNR range [min, max] for triggering recalculation. Default is [6,8].

       **snr_recalculation_waveform_approximant** : `str`
           Waveform approximant to use for SNR recalculation. Default is 'IMRPhenomXPHM'.











   .. rubric:: Examples

   >>> from gwsnr import GWSNR
   >>> snr = GWSNR()
   >>> snroptimal_snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)

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
   |:attr:`~spin_max`                    | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~mtot_resolution`             | `int`                            |
   +-------------------------------------+----------------------------------+
   |:attr:`~ratio_resolution`            | `int`                            |
   +-------------------------------------+----------------------------------+
   |:attr:`~spin_resolution`             | `int`                            |
   +-------------------------------------+----------------------------------+
   |:attr:`~ratio_arr`                   | `numpy.ndarray`                  |
   +-------------------------------------+----------------------------------+
   |:attr:`~mtot_arr`                    | `numpy.ndarray`                  |
   +-------------------------------------+----------------------------------+
   |:attr:`~a_1_arr`                     | `numpy.ndarray`                  |
   +-------------------------------------+----------------------------------+
   |:attr:`~a_2_arr`                     | `numpy.ndarray`                  |
   +-------------------------------------+----------------------------------+
   |:attr:`~sampling_frequency`          | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~waveform_approximant`        | `str`                            |
   +-------------------------------------+----------------------------------+
   |:attr:`~frequency_domain_source_model`| `str`                           |
   +-------------------------------------+----------------------------------+
   |:attr:`~f_min`                       | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~f_ref`                       | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~duration_max`                | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~duration_min`                | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~snr_method`                    | `str`                            |
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
   |:attr:`~snr_partialsacaled_list`     | `list` of `numpy.ndarray`        |
   +-------------------------------------+----------------------------------+
   |:attr:`~multiprocessing_verbose`     | `bool`                           |
   +-------------------------------------+----------------------------------+
   |:attr:`~param_dict_given`            | `dict`                           |
   +-------------------------------------+----------------------------------+
   |:attr:`~pdet`                        | `bool` or `str`                  |
   +-------------------------------------+----------------------------------+
   |:attr:`~snr_th`                      | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~snr_th_net`                  | `float`                          |
   +-------------------------------------+----------------------------------+
   |:attr:`~model_dict`                  | `dict` (ANN models)              |
   +-------------------------------------+----------------------------------+
   |:attr:`~scaler_dict`                 | `dict` (ANN scalers)             |
   +-------------------------------------+----------------------------------+
   |:attr:`~error_adjustment`            | `dict` (ANN error correction)    |
   +-------------------------------------+----------------------------------+
   |:attr:`~ann_catalogue`               | `dict` (ANN configuration)       |
   +-------------------------------------+----------------------------------+
   |:attr:`~snr_recalculation`           | `bool`                           |
   +-------------------------------------+----------------------------------+
   |:attr:`~snr_recalculation_range`     | `list`                           |
   +-------------------------------------+----------------------------------+
   |:attr:`~snr_recalculation_waveform_approximant`| `str`               |
   +-------------------------------------+----------------------------------+

   Instance Methods
   ----------
   GWSNR class has the following methods,

   +-------------------------------------+----------------------------------+
   | Methods                             | Description                      |
   +=====================================+==================================+
   |:meth:`~snr`                         | Main method that calls           |
   |                                     | appropriate SNR calculation      |
   |                                     | based on :attr:`~snr_method`.      |
   +-------------------------------------+----------------------------------+
   |:meth:`~optimal_snr_with_interpolation`      | Calculates SNR using             |
   |                                     | interpolation method.            |
   +-------------------------------------+----------------------------------+
   |:meth:`~optimal_snr_with_ann`                | Calculates SNR using             |
   |                                     | artificial neural network.       |
   +-------------------------------------+----------------------------------+
   |:meth:`~optimal_snr_with_inner_product`           | Calculates SNR using             |
   |                                     | inner product method             |
   |                                     | (python multiprocessing).        |
   +-------------------------------------+----------------------------------+
   |:meth:`~optimal_snr_with_inner_product_ripple`          | Calculates SNR using             |
   |                                     | inner product method             |
   |                                     | (jax.jit+jax.vmap).              |
   +-------------------------------------+----------------------------------+
   |:meth:`~horizon_distance`            | Calculates detector horizon      |
   |                                     | distance.                        |
   +-------------------------------------+----------------------------------+
   |:meth:`~pdet`    | Calculates probability of        |
   |                                     | detection.                       |
   +-------------------------------------+----------------------------------+
   |:meth:`~print_all_params`            | Prints all the parameters of     |
   |                                     | the class instance.              |
   +-------------------------------------+----------------------------------+
   |:meth:`~init_partialscaled`          | Generates partialscaled SNR      |
   |                                     | interpolation coefficients.      |
   +-------------------------------------+----------------------------------+
   |:meth:`~interpolator_setup`          | Sets up interpolator files       |
   |                                     | and handles caching.             |
   +-------------------------------------+----------------------------------+
   |:meth:`~ann_initilization`           | Initializes ANN models and       |
   |                                     | scalers for detection.           |
   +-------------------------------------+----------------------------------+
   |:meth:`~output_ann`                  | Prepares input features for      |
   |                                     | ANN prediction.                  |
   +-------------------------------------+----------------------------------+
   |:meth:`~calculate_mtot_max`          | Calculates maximum total mass    |
   |                                     | based on minimum frequency.      |
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

   .. py:attribute:: spin_max

      
      ``float``

      Maximum spin magnitude for aligned-spin interpolation methods.















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

   .. py:attribute:: spin_resolution

      
      ``int``

      Number of points in the spin arrays for aligned-spin interpolation methods.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_arr

      
      ``numpy.ndarray``

      Array of mass ratio.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_arr

      
      ``numpy.ndarray``

      Array of total mass.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: a_1_arr

      
      ``numpy.ndarray``

      Array of primary spin values for aligned-spin interpolation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: a_2_arr

      
      ``numpy.ndarray``

      Array of secondary spin values for aligned-spin interpolation.















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

   .. py:attribute:: frequency_domain_source_model

      
      ``str``

      Source model for frequency domain waveform generation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_min

      
      ``float``

      Minimum frequency of the waveform.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_ref

      
      ``float``

      Reference frequency of the waveform.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: duration_max

      
      ``float`` or ``None``

      Maximum duration for waveform generation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: duration_min

      
      ``float`` or ``None``

      Minimum duration for waveform generation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_method

      
      ``str``

      Type of SNR calculation. Options: 'interpolation', 'interpolation_no_spins', 'interpolation_no_spins_jax', 'interpolation_no_spins_mlx', 'interpolation_aligned_spins', 'interpolation_aligned_spins_jax', 'interpolation_aligned_spins_mlx', 'inner_product', 'inner_product_jax', 'ann'.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: psds_list

      
      ``list`` of bilby's PowerSpectralDensity ``object``

      List of power spectral density objects for different detectors.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: detector_tensor_list

      
      ``list`` of detector tensor ``numpy.ndarray``

      List of detector tensor arrays for antenna response calculations.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: detector_list

      
      ``list`` of ``str``

      List of detector names.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ifos

      
      ``list`` of bilby's Interferometer ``object``

      Bilby interferometer objects for the detectors.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: interpolator_dir

      
      ``str``

      Path to store the interpolator pickle file.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: path_interpolator

      
      ``list`` of ``str``

      List of paths to interpolator pickle files for each detector.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_partialsacaled_list

      
      ``list`` of ``numpy.ndarray``

      List of partial-scaled SNR interpolation coefficients for each detector.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: multiprocessing_verbose

      
      ``bool``

      If True, show progress bar during SNR computation with multiprocessing.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: param_dict_given

      
      ``dict``

      Dictionary containing interpolator parameters for identification and caching.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: pdet

      
      ``bool`` or ``str``

      If True or 'bool', calculate probability of detection using boolean method. If 'matched_filter', use matched filter probability. Default is False.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_th

      
      ``float``

      SNR threshold for individual detector for pdet calculation. Default is 8.0.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_th_net

      
      ``float``

      SNR threshold for network SNR for pdet calculation. Default is 8.0.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: model_dict

      
      ``dict``

      Dictionary of ANN models for different detectors (used when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: scaler_dict

      
      ``dict``

      Dictionary of ANN feature scalers for different detectors (used when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: error_adjustment

      
      ``dict``

      Dictionary of ANN error correction parameters for different detectors (used when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ann_catalogue

      
      ``dict``

      Dictionary containing ANN configuration and model paths (used when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_recalculation

      
      ``bool``

      If True, enables hybrid SNR recalculation for systems near detection threshold.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_recalculation_range

      
      ``list``

      SNR range [min, max] for triggering recalculation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_recalculation_waveform_approximant

      
      ``str``

      Waveform approximant to use for SNR recalculation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: get_interpolated_snr

      
      ``function``

      Function for interpolated SNR calculation (set based on snr_method).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: noise_weighted_inner_product_jax

      
      ``function``

      JAX-accelerated noise-weighted inner product function (used when snr_method='inner_product_jax').















      ..
          !! processed by numpydoc !!

   .. py:method:: interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

      
      Function to set up interpolator files and handle caching for partialscaled SNR interpolation coefficients.

      This method checks for existing interpolator files, determines which detectors need new interpolators,
      and manages the generation and loading of partialscaled SNR interpolation data. It handles both
      the creation of new interpolators and the loading of existing ones from cache.

      :Parameters:

          **interpolator_dir** : `str`
              Path to directory for storing interpolator pickle files. Default is './interpolator_pickle'.

          **create_new_interpolator** : `bool`
              If True, forces generation of new interpolators or replaces existing ones. If False,
              uses existing interpolators when available. Default is False.

          **psds_list** : `list` of bilby's PowerSpectralDensity objects
              List of power spectral density objects for different detectors used for interpolator generation.

          **detector_tensor_list** : `list` of `numpy.ndarray`
              List of detector tensor arrays for antenna response calculations during interpolator generation.

          **detector_list** : `list` of `str`
              List of detector names (e.g., ['L1', 'H1', 'V1']) for which interpolators are needed.

      :Returns:

          **path_interpolator_all** : `list` of `str`
              List of file paths to partialscaled SNR interpolator pickle files for all detectors.
              These files contain the precomputed interpolation coefficients used for fast SNR calculation.








      .. rubric:: Notes

      - The method uses :func:`~self.utils.interpolator_check` to determine which detectors need new interpolators
      - For missing interpolators, calls :meth:`~init_partialscaled` to generate them
      - Updates class attributes including :attr:`~psds_list`, :attr:`~detector_tensor_list`, :attr:`~detector_list`, and :attr:`~path_interpolator`
      - Loads all interpolator data into :attr:`~snr_partialsacaled_list` for runtime use
      - Supports both no-spin and aligned-spin interpolation methods based on :attr:`~snr_method`





      ..
          !! processed by numpydoc !!

   .. py:method:: ann_initilization(ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant, snr_th)

      
      Function to initialize ANN models and scalers for detection probability estimation using artificial neural networks.

      This method loads and validates ANN models, feature scalers, and error correction parameters for each detector
      in the detector list. It handles both built-in models from the gwsnr package and user-provided models,
      ensuring compatibility with the current GWSNR configuration parameters.

      :Parameters:

          **ann_path_dict** : `dict` or `str` or `None`
              Dictionary or path to JSON file containing ANN model and scaler paths for different detectors.
              If None, uses default models from gwsnr/ann/data/ann_path_dict.json.
              If dict, should have structure: {detector_name: {'model_path': str, 'scaler_path': str,
              'error_adjustment_path': str, 'sampling_frequency': float, 'minimum_frequency': float,
              'waveform_approximant': str, 'snr_th': float}}.

          **detector_list** : `list` of `str`
              List of detector names (e.g., ['L1', 'H1', 'V1']) for which ANN models are needed.

          **sampling_frequency** : `float`
              Sampling frequency of the detector data. Must match ANN training parameters.

          **minimum_frequency** : `float`
              Minimum frequency of the waveform. Must match ANN training parameters.

          **waveform_approximant** : `str`
              Waveform approximant to use. Must match ANN training parameters.

          **snr_th** : `float`
              SNR threshold for individual detector detection. Must match ANN training parameters.

      :Returns:

          **model_dict** : `dict`
              Dictionary of loaded ANN models for each detector {detector_name: tensorflow.keras.Model}.

          **scaler_dict** : `dict`
              Dictionary of loaded feature scalers for each detector {detector_name: sklearn.preprocessing.Scaler}.

          **error_adjustment** : `dict`
              Dictionary of error correction parameters for each detector {detector_name: {'slope': float, 'intercept': float}}.

          **ann_catalogue** : `dict`
              Dictionary containing complete ANN configuration and model paths for all detectors.




      :Raises:

          ValueError
              If ANN model or scaler is not available for a detector in detector_list.
              If model parameters don't match the current GWSNR configuration.
              If required keys ('model_path', 'scaler_path') are missing from ann_path_dict.




      .. rubric:: Notes

      - Models are loaded from gwsnr/ann/data directory if paths don't exist as files
      - Parameter validation ensures ANN models are compatible with current settings
      - Error adjustment parameters provide post-prediction correction for improved accuracy
      - ANN models use partial-scaled SNR as input feature along with other parameters





      ..
          !! processed by numpydoc !!

   .. py:method:: calculate_mtot_max(mtot_max, minimum_frequency)

      
      Function to calculate the maximum total mass cutoff based on minimum frequency to ensure positive chirp time.

      This method determines the maximum allowable total mass for binary systems by finding where
      the chirp time becomes zero at the given minimum frequency. The chirp time represents the
      duration a gravitational wave signal spends in the detector's frequency band. A safety factor
      of 1.1 is applied to ensure the chirp time remains positive for waveform generation.

      The calculation uses the :func:`~self.numba.findchirp_chirptime` function to compute chirp
      times and employs numerical root finding to determine where the chirp time approaches zero.

      :Parameters:

          **mtot_max** : `float`
              User-specified maximum total mass of the binary in solar masses. If this exceeds
              the frequency-based limit, it will be reduced to the calculated maximum.

          **minimum_frequency** : `float`
              Minimum frequency of the waveform in Hz. Lower frequencies allow higher total masses
              before the chirp time becomes negative.

      :Returns:

          **mtot_max** : `float`
              Adjusted maximum total mass of the binary in solar masses, ensuring positive chirp
              time at the given minimum frequency. Will be the smaller of the input mtot_max and
              the frequency-based limit.








      .. rubric:: Notes

      - Uses equal mass ratio (q=1.0) for the chirp time calculation as a conservative estimate
      - The safety factor of 1.1 provides a buffer to prevent numerical issues during waveform generation
      - This limit is particularly important for low-frequency detectors and TaylorF2 approximants
      - The method uses :func:`scipy.optimize.fsolve` to find the root of the chirp time function





      ..
          !! processed by numpydoc !!

   .. py:method:: print_all_params(verbose=True)

      
      Function to print all the parameters and configuration of the GWSNR class instance.

      This method displays comprehensive information about the current GWSNR configuration including
      computational parameters, waveform settings, detector configuration, interpolation grid parameters,
      and file paths. It provides a complete overview of the initialized GWSNR instance for verification
      and debugging purposes.

      :Parameters:

          **verbose** : `bool`
              If True, print all the parameters of the class instance to stdout. If False,
              suppress output. Default is True.









      .. rubric:: Notes

      The printed information includes:

      - **Computational settings**: Number of processors (:attr:`~npool`), SNR calculation type (:attr:`~snr_method`)
      - **Waveform configuration**: Approximant (:attr:`~waveform_approximant`), sampling frequency (:attr:`~sampling_frequency`), minimum frequency (:attr:`~f_min`)
      - **Mass parameter ranges**: Total mass bounds (:attr:`~mtot_min`, :attr:`~mtot_max`) with frequency-based cutoff information
      - **Detector setup**: List of detectors (:attr:`~detector_list`) and their PSDs (:attr:`~psds_list`)
      - **Interpolation parameters**: Mass ratio bounds (:attr:`~ratio_min`, :attr:`~ratio_max`), grid resolutions (:attr:`~mtot_resolution`, :attr:`~ratio_resolution`)
      - **File paths**: Interpolator directory (:attr:`~interpolator_dir`) when using interpolation methods

      This method is automatically called during class initialization when :attr:`~gwsnr_verbose` is True.


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> snr = GWSNR(gwsnr_verbose=False)  # Initialize without printing
      >>> snr.print_all_params()  # Manually print parameters



      ..
          !! processed by numpydoc !!

   .. py:method:: snr(mass_1=np.array([10.0]), mass_2=np.array([10.0]), luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Main function to calculate SNR of gravitational-wave signals from compact binary coalescences.

      This method serves as the primary interface for SNR calculation, automatically routing to the
      appropriate computation method based on the :attr:`~snr_method` setting. It supports multiple
      backend methods including interpolation-based fast calculation, inner product methods, JAX-accelerated
      computation, and artificial neural network estimation.

      The method handles parameter validation, coordinate transformations (e.g., tilt angles to aligned spins),
      and optionally computes probability of detection. For systems near detection threshold, it can perform
      hybrid SNR recalculation using more accurate waveform models.

      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar masses. Default is np.array([10.0]).

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar masses. Default is np.array([10.0]).

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Gravitational wave polarization angle in radians. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Gravitational wave phase at coalescence in radians. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the source in radians. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the source in radians. Default is 0.0.

          **a_1** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the primary object. Default is 0.0.

          **a_2** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the secondary object. Default is 0.0.

          **tilt_1** : `numpy.ndarray` or `float`
              Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.

          **tilt_2** : `numpy.ndarray` or `float`
              Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.

          **phi_12** : `numpy.ndarray` or `float`
              Azimuthal angle between the two spins in radians. Default is 0.0.

          **phi_jl** : `numpy.ndarray` or `float`
              Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.

          **lambda_1** : `numpy.ndarray` or `float`
              Dimensionless tidal deformability of primary object. Default is 0.0.

          **lambda_2** : `numpy.ndarray` or `float`
              Dimensionless tidal deformability of secondary object. Default is 0.0.

          **eccentricity** : `numpy.ndarray` or `float`
              Orbital eccentricity at reference frequency. Default is 0.0.

          **gw_param_dict** : `dict` or `bool`
              Dictionary containing all gravitational wave parameters as key-value pairs.
              If provided, takes precedence over individual parameter arguments. Default is False.

          **output_jsonfile** : `str` or `bool`
              If string, saves the SNR results to a JSON file with the given filename.
              If True, saves to 'snr.json'. If False, no file output. Default is False.

      :Returns:

          **snr_dict** : `dict`
              Dictionary containing SNR values for each detector and network SNR.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are numpy arrays of SNR values corresponding to input parameters.

          **pdet_dict** : `dict`
              Dictionary containing probability of detection values (only if :attr:`~pdet` is True).
              Keys include detector names and 'pdet_net'. Values are numpy arrays of probabilities.




      :Raises:

          ValueError
              If :attr:`~snr_method` is not recognized or if parameters are outside valid ranges.




      .. rubric:: Notes

      - For interpolation methods, aligned spin components are computed as a_i * cos(tilt_i)
      - Total mass must be within [mtot_min, mtot_max] range for interpolation methods
      - Hybrid SNR recalculation is triggered when :attr:`~snr_recalculation` is True and
        network SNR falls within :attr:`~snr_recalculation_range`
      - When :attr:`~pdet` is True, returns detection probabilities instead of SNR values


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> # Basic interpolation-based SNR calculation
      >>> snr = GWSNR(snr_method='interpolation')
      >>> result = snroptimal_snr(mass_1=30.0, mass_2=30.0, luminosity_distance=100.0)

      >>> # Using parameter dictionary
      >>> params = {'mass_1': [20, 30], 'mass_2': [20, 30], 'luminosity_distance': [100, 200]}
      >>> result = snroptimal_snr(gw_param_dict=params)

      >>> # With probability of detection
      >>> snr_pdet = GWSNR(snr_method='interpolation', pdet=True)
      >>> pdet_result = snr_pdetoptimal_snr(mass_1=30.0, mass_2=30.0, luminosity_distance=100.0)



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_ann(mass_1=30.0, mass_2=29.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Function to calculate SNR using artificial neural network (ANN) estimation method.

      This method uses trained neural networks to rapidly estimate the probability of detection (Pdet)
      for spin-precessing gravitational wave signals. The ANN models leverage partial-scaled SNR as a
      summary statistic along with other intrinsic parameters to provide fast detection probability
      estimates, particularly useful for population synthesis studies.

      The method first calculates partial-scaled SNR using interpolation, then uses this as input
      to pre-trained ANN models for each detector. Error correction is applied to improve accuracy
      of the ANN predictions.

      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar masses. Default is 30.0.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar masses. Default is 29.0.

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Gravitational wave polarization angle in radians. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Gravitational wave phase at coalescence in radians. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the source in radians. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the source in radians. Default is 0.0.

          **a_1** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the primary object. Default is 0.0.

          **a_2** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the secondary object. Default is 0.0.

          **tilt_1** : `numpy.ndarray` or `float`
              Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.

          **tilt_2** : `numpy.ndarray` or `float`
              Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.

          **phi_12** : `numpy.ndarray` or `float`
              Azimuthal angle between the two spins in radians. Default is 0.0.

          **phi_jl** : `numpy.ndarray` or `float`
              Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.

          **gw_param_dict** : `dict` or `bool`
              Dictionary containing all gravitational wave parameters as key-value pairs.
              If provided, takes precedence over individual parameter arguments. Default is False.

          **output_jsonfile** : `str` or `bool`
              If string, saves the SNR results to a JSON file with the given filename.
              If True, saves to 'snr.json'. If False, no file output. Default is False.

      :Returns:

          **optimal_snr** : `dict`
              Dictionary containing ANN-estimated SNR values for each detector and network SNR.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are numpy arrays of SNR estimates corresponding to input parameters.




      :Raises:

          ValueError
              If total mass (mass_1 + mass_2) is outside the range [mtot_min, mtot_max].




      .. rubric:: Notes

      - ANN models must be pre-trained and loaded during class initialization
      - Uses aligned spin components calculated as a_i * cos(tilt_i) for feature input
      - Feature inputs include: partial-scaled SNR, amplitude factor, symmetric mass ratio,
        effective spin, and inclination angle
      - Error adjustment parameters provide post-prediction correction for improved accuracy
      - Compatible with waveform approximants that have corresponding trained ANN models
      - Requires :attr:`~snr_method` to be set to 'ann' during GWSNR initialization


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> # Initialize with ANN method
      >>> snr = GWSNR(snr_method='ann', waveform_approximant='IMRPhenomXPHM')
      >>> # Calculate SNR using ANN
      >>> result = snr.optimal_snr_with_ann(mass_1=30.0, mass_2=25.0, luminosity_distance=200.0,
      ...                          a_1=0.5, a_2=0.3, tilt_1=0.2, tilt_2=0.1)

      >>> # Using parameter dictionary
      >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200],
      ...           'a_1': [0.2, 0.5], 'tilt_1': [0.1, 0.3]}
      >>> result = snr.optimal_snr_with_ann(gw_param_dict=params)



      ..
          !! processed by numpydoc !!

   .. py:method:: output_ann(idx, params)

      
      Function to prepare input features for ANN prediction from gravitational wave parameters.

      This method transforms gravitational wave parameters into feature vectors suitable for
      artificial neural network prediction of detection probabilities. It calculates partial-scaled
      SNR using interpolation and combines it with other intrinsic parameters to create the input
      features expected by the pre-trained ANN models.

      The feature vector for each detector includes:
      - Partial-scaled SNR (dimensionless, distance-independent)
      - Amplitude factor (A1 = Mc^(5/6) / d_eff)
      - Symmetric mass ratio (eta)
      - Effective spin (chi_eff)
      - Inclination angle (theta_jn)

      :Parameters:

          **idx** : `numpy.ndarray` of `bool`
              Boolean index array indicating which parameter entries are within valid mass ranges
              for interpolation (mtot_min <= mtot <= mtot_max).

          **params** : `dict`
              Dictionary containing gravitational wave parameters with keys:
              - 'mass_1', 'mass_2': Primary and secondary masses in solar masses
              - 'luminosity_distance': Distance in Mpc
              - 'theta_jn': Inclination angle in radians
              - 'a_1', 'a_2': Spin magnitudes (dimensionless)
              - 'tilt_1', 'tilt_2': Spin tilt angles in radians
              - 'psi', 'geocent_time', 'ra', 'dec': Extrinsic parameters

      :Returns:

          **ann_input** : `list` of `numpy.ndarray`
              List of feature arrays for each detector in :attr:`~detector_list`.
              Each array has shape (N, 5) where N is the number of valid parameter sets,
              and columns correspond to [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn].








      .. rubric:: Notes

      - Uses :meth:`~get_interpolated_snr` to calculate partial-scaled SNR via interpolation
      - Aligned spin components are computed as a_i * cos(tilt_i) for chi_eff calculation
      - Chirp mass Mc = (m1*m2)^(3/5) / (m1+m2)^(1/5) is used for amplitude scaling
      - Effective spin chi_eff = (m1*a1z + m2*a2z) / (m1+m2) where aiz are aligned components
      - Feature scaling is applied later using pre-loaded scalers in :meth:`~optimal_snr_with_ann`





      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_interpolation(mass_1=30.0, mass_2=29.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, output_jsonfile=False, gw_param_dict=False)

      
      Function to calculate SNR using bicubic interpolation of precomputed partial-scaled SNR coefficients.

      This method provides fast SNR calculation by interpolating precomputed partial-scaled SNR values
      across a grid of intrinsic parameters (total mass, mass ratio, and optionally aligned spins).
      The interpolation is performed using either Numba-accelerated or JAX-accelerated functions
      depending on the :attr:`~snr_method` setting. This approach is particularly efficient for
      large-scale population studies and parameter estimation.

      The method handles parameter validation, ensures masses are within interpolation bounds,
      and computes antenna response patterns for each detector. For systems outside the mass
      range, SNR values are set to zero.

      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar masses. Default is 30.0.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar masses. Default is 29.0.

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Gravitational wave polarization angle in radians. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Gravitational wave phase at coalescence in radians. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the source in radians. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the source in radians. Default is 0.0.

          **a_1** : `numpy.ndarray` or `float`
              Dimensionless aligned spin component of the primary object (only used for aligned-spin interpolation types). Default is 0.0.

          **a_2** : `numpy.ndarray` or `float`
              Dimensionless aligned spin component of the secondary object (only used for aligned-spin interpolation types). Default is 0.0.

          **gw_param_dict** : `dict` or `bool`
              Dictionary containing all gravitational wave parameters as key-value pairs.
              If provided, takes precedence over individual parameter arguments. Default is False.

          **output_jsonfile** : `str` or `bool`
              If string, saves the SNR results to a JSON file with the given filename.
              If True, saves to 'snr.json'. If False, no file output. Default is False.

      :Returns:

          **optimal_snr** : `dict`
              Dictionary containing SNR values for each detector and network SNR.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are numpy arrays of SNR values corresponding to input parameters.
              Systems with total mass outside [mtot_min, mtot_max] have SNR set to zero.








      .. rubric:: Notes

      - Requires precomputed interpolation coefficients stored in :attr:`~snr_partialsacaled_list`
      - Total mass (mass_1 + mass_2) must be within [mtot_min, mtot_max] for non-zero SNR
      - For aligned-spin methods, uses aligned spin components computed as a_i * cos(tilt_i)
      - Interpolation grid parameters are set during class initialization
      - Supports both Numba and JAX backends for accelerated computation
      - Compatible with waveform approximants: IMRPhenomD, TaylorF2, IMRPhenomXPHM


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> # No-spin interpolation
      >>> snr = GWSNR(snr_method='interpolation_no_spins')
      >>> result = snr.optimal_snr_with_interpolation(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)

      >>> # Aligned-spin interpolation
      >>> snr_spin = GWSNR(snr_method='interpolation_aligned_spins')
      >>> result = snr_spin.optimal_snr_with_interpolation(mass_1=30.0, mass_2=25.0,
      ...                                         luminosity_distance=100.0, a_1=0.5, a_2=0.3)

      >>> # Using parameter dictionary
      >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
      >>> result = snr.optimal_snr_with_interpolation(gw_param_dict=params)



      ..
          !! processed by numpydoc !!

   .. py:method:: init_partialscaled()

      
      Function to generate partialscaled SNR interpolation coefficients for fast bicubic interpolation.

      This method computes and saves precomputed partial-scaled SNR values across a grid of intrinsic
      parameters (total mass, mass ratio, and optionally aligned spins) for each detector in the network.
      The partial-scaled SNR is distance-independent and decoupled from extrinsic parameters, enabling
      fast interpolation during runtime SNR calculations.

      The method creates a parameter grid based on the interpolation type:
      - For no-spin methods: 2D grid over (mass_ratio, total_mass)
      - For aligned-spin methods: 4D grid over (mass_ratio, total_mass, a_1, a_2)

      For each grid point, it computes the optimal SNR using :meth:`~optimal_snr_with_inner_product` with fixed
      extrinsic parameters, then scales by effective luminosity distance and chirp mass to create
      the partial-scaled SNR coefficients. These coefficients are saved as pickle files for later
      use during interpolation-based SNR calculations.

      :Parameters:

          **None**
              Uses class attributes for grid parameters and detector configuration.

      :Returns:

          None
              Saves interpolation coefficients to pickle files specified in :attr:`~path_interpolator`.




      :Raises:

          ValueError
              If :attr:`~mtot_min` is less than 1.0 solar mass.
              If :attr:`~snr_method` is not supported for interpolation.




      .. rubric:: Notes

      - Uses fixed extrinsic parameters: luminosity_distance=100 Mpc, theta_jn=0, ra=0, dec=0, psi=0, phase=0
      - Calls :meth:`~optimal_snr_with_inner_product` to generate unscaled SNR values across the parameter grid
      - Partial-scaled SNR = (optimal_SNR * d_eff) / Mc^(5/6) where Mc is chirp mass
      - Grid dimensions depend on resolution parameters: :attr:`~ratio_resolution`, :attr:`~mtot_resolution`, :attr:`~spin_resolution`
      - For aligned-spin methods, grid covers spin range [-spin_max, +spin_max] for both objects
      - Interpolation coefficients enable fast runtime SNR calculation via bicubic interpolation
      - Compatible with snr_methods: 'interpolation', 'interpolation_no_spins', 'interpolation_aligned_spins', and their JAX variants


      .. rubric:: Examples

      This method is called automatically during GWSNR initialization when interpolation
      coefficients don't exist or when :attr:`~create_new_interpolator` is True.

      >>> from gwsnr import GWSNR
      >>> # Will automatically call init_partialscaled() if needed
      >>> snr = GWSNR(snr_method='interpolation_no_spins', create_new_interpolator=True)



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_inner_product(mass_1=10, mass_2=10, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Function to calculate SNR using noise-weighted inner product method with LAL waveform generation.

      This method computes the optimal signal-to-noise ratio using the standard matched filtering
      formalism with noise-weighted inner products between gravitational wave signals and detector
      noise power spectral densities. It supports multiprocessing for efficient computation and
      is compatible with various waveform approximants from LALSimulation.

      The method generates frequency-domain waveforms using LAL, computes the inner products
      with detector PSDs, and calculates antenna response patterns for each detector in the
      network. It automatically handles duration estimation based on chirp time and supports
      systems with arbitrary spin configurations including precession.

      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar masses. Default is 10.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar masses. Default is 10.

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Gravitational wave polarization angle in radians. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Gravitational wave phase at coalescence in radians. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the source in radians. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the source in radians. Default is 0.0.

          **a_1** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the primary object. Default is 0.0.

          **a_2** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the secondary object. Default is 0.0.

          **tilt_1** : `numpy.ndarray` or `float`
              Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.

          **tilt_2** : `numpy.ndarray` or `float`
              Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.

          **phi_12** : `numpy.ndarray` or `float`
              Azimuthal angle between the two spins in radians. Default is 0.0.

          **phi_jl** : `numpy.ndarray` or `float`
              Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.

          **lambda_1** : `numpy.ndarray` or `float`
              Dimensionless tidal deformability of primary object. Default is 0.0.

          **lambda_2** : `numpy.ndarray` or `float`
              Dimensionless tidal deformability of secondary object. Default is 0.0.

          **eccentricity** : `numpy.ndarray` or `float`
              Orbital eccentricity at reference frequency. Default is 0.0.

          **gw_param_dict** : `dict` or `bool`
              Dictionary containing all gravitational wave parameters as key-value pairs.
              If provided, takes precedence over individual parameter arguments. Default is False.

          **output_jsonfile** : `str` or `bool`
              If string, saves the SNR results to a JSON file with the given filename.
              If True, saves to 'snr.json'. If False, no file output. Default is False.

      :Returns:

          **optimal_snr** : `dict`
              Dictionary containing SNR values for each detector and network SNR.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are numpy arrays of SNR values corresponding to input parameters.
              Systems with total mass outside [mtot_min, mtot_max] have SNR set to zero.








      .. rubric:: Notes

      - Uses LALSimulation for frequency-domain waveform generation
      - Automatically estimates waveform duration based on chirp time with safety factor
      - Duration is bounded by :attr:`~duration_min` and :attr:`~duration_max` if specified
      - Supports multiprocessing with :attr:`~npool` processors for parallel computation
      - Compatible with all LAL waveform approximants including precessing and higher-order modes
      - Uses :func:`~self.utils.noise_weighted_inner_prod` for inner product calculation
      - Antenna response patterns computed using :func:`~self.numba.antenna_response_array`


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> # Initialize with inner product method
      >>> snr = GWSNR(snr_method='inner_product')
      >>> # Calculate SNR for aligned systems
      >>> result = snr.optimal_snr_with_inner_product(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)

      >>> # Calculate SNR for precessing systems
      >>> result = snr.optimal_snr_with_inner_product(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0,
      ...                               a_1=0.5, a_2=0.3, tilt_1=0.2, tilt_2=0.1)

      >>> # Using parameter dictionary
      >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
      >>> result = snr.optimal_snr_with_inner_product(gw_param_dict=params)



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_inner_product_ripple(mass_1=10, mass_2=10, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Function to calculate SNR using JAX-accelerated noise-weighted inner product method with Ripple waveform generation.

      This method computes the optimal signal-to-noise ratio using JAX-accelerated inner products between
      gravitational wave signals generated with the Ripple waveform generator and detector noise power
      spectral densities. It leverages JAX's just-in-time (JIT) compilation and vectorized map (vmap)
      functions for highly efficient parallelized computation, making it suitable for large-scale
      parameter estimation and population studies.

      The method uses the RippleInnerProduct class for waveform generation and inner product calculation,
      automatically handling duration estimation and supporting arbitrary spin configurations. It provides
      significant computational speedup compared to traditional LAL-based methods while maintaining
      numerical accuracy.

      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar masses. Default is 10.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar masses. Default is 10.

          **luminosity_distance** : `numpy.ndarray` or `float`
              Luminosity distance of the binary in Mpc. Default is 100.0.

          **theta_jn** : `numpy.ndarray` or `float`
              Inclination angle between total angular momentum and line of sight in radians. Default is 0.0.

          **psi** : `numpy.ndarray` or `float`
              Gravitational wave polarization angle in radians. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Gravitational wave phase at coalescence in radians. Default is 0.0.

          **geocent_time** : `numpy.ndarray` or `float`
              GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.

          **ra** : `numpy.ndarray` or `float`
              Right ascension of the source in radians. Default is 0.0.

          **dec** : `numpy.ndarray` or `float`
              Declination of the source in radians. Default is 0.0.

          **a_1** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the primary object. Default is 0.0.

          **a_2** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the secondary object. Default is 0.0.

          **tilt_1** : `numpy.ndarray` or `float`
              Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.

          **tilt_2** : `numpy.ndarray` or `float`
              Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0.0.

          **phi_12** : `numpy.ndarray` or `float`
              Azimuthal angle between the two spins in radians. Default is 0.0.

          **phi_jl** : `numpy.ndarray` or `float`
              Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.

          **gw_param_dict** : `dict` or `bool`
              Dictionary containing all gravitational wave parameters as key-value pairs.
              If provided, takes precedence over individual parameter arguments. Default is False.

          **output_jsonfile** : `str` or `bool`
              If string, saves the SNR results to a JSON file with the given filename.
              If True, saves to 'snr.json'. If False, no file output. Default is False.

      :Returns:

          **optimal_snr** : `dict`
              Dictionary containing SNR values for each detector and network SNR.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are numpy arrays of SNR values corresponding to input parameters.
              Systems with total mass outside [mtot_min, mtot_max] have SNR set to zero.








      .. rubric:: Notes

      - Uses Ripple waveform generator with JAX backend for GPU acceleration
      - Automatically estimates waveform duration bounded by :attr:`~duration_min` and :attr:`~duration_max`
      - Compatible with waveform approximants supported by Ripple (e.g., IMRPhenomD, IMRPhenomXPHM)
      - Leverages JAX's jit and vmap for vectorized batch processing
      - Supports multiprocessing with :attr:`~npool` processors when applicable
      - Uses :meth:`~RippleInnerProduct.noise_weighted_inner_product_jax` for inner product calculation
      - Antenna response patterns computed using :func:`~self.numba.antenna_response_array`
      - Requires :attr:`~snr_method` to be set to 'inner_product_jax' during GWSNR initialization


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> # Initialize with JAX inner product method
      >>> snr = GWSNR(snr_method='inner_product_jax', waveform_approximant='IMRPhenomD')
      >>> # Calculate SNR for aligned systems
      >>> result = snr.optimal_snr_with_inner_product_ripple(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)

      >>> # Calculate SNR for precessing systems
      >>> result = snr.optimal_snr_with_inner_product_ripple(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0,
      ...                                a_1=0.5, a_2=0.3, tilt_1=0.2, tilt_2=0.1)

      >>> # Using parameter dictionary
      >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
      >>> result = snr.optimal_snr_with_inner_product_ripple(gw_param_dict=params)



      ..
          !! processed by numpydoc !!

   .. py:method:: pdet(snr_dict, snr_th=None, snr_th_net=None, type='matched_filter')

      
      Function to calculate probability of detection for gravitational wave signals using SNR threshold criteria.

      This method computes the probability of detection (Pdet) for gravitational wave signals based on
      signal-to-noise ratio thresholds for individual detectors and the detector network. It supports
      both matched filter probability calculation using Gaussian noise assumptions and simple boolean
      threshold detection. The method is compatible with single or multiple SNR threshold values for
      different detectors in the network.

      :Parameters:

          **snr_dict** : `dict`
              Dictionary containing SNR values for each detector and network SNR.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are numpy arrays of SNR values corresponding to input parameters.

          **snr_th** : `float` or `numpy.ndarray` or `None`
              SNR threshold for individual detector detection. If None, uses :attr:`~snr_th`.
              If array, must have length equal to number of detectors. Default is None.

          **snr_th_net** : `float` or `None`
              SNR threshold for network detection. If None, uses :attr:`~snr_th_net`. Default is None.

          **type** : `str`
              Type of probability calculation method. Default is 'matched_filter'.
              Options: 'matched_filter' (Gaussian noise probability), 'bool' (boolean threshold).

      :Returns:

          **pdet_dict** : `dict`
              Dictionary containing probability of detection for each detector and network.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'pdet_net'.
              Values are numpy arrays of detection probabilities [0,1] for 'matched_filter'
              or boolean arrays {0,1} for 'bool' type.








      .. rubric:: Notes

      - For 'matched_filter' type: Uses Gaussian noise assumption with Pdet = 1 - Φ(ρ_th - ρ)
        where Φ is the cumulative distribution function of the standard normal distribution
      - For 'bool' type: Returns 1 if SNR > threshold, 0 otherwise
      - Individual detector thresholds can be different by providing array of thresholds
      - Network detection uses quadrature sum of individual detector SNRs
      - Compatible with all SNR calculation methods (interpolation, inner product, ANN)


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> snr = GWSNR(snr_method='interpolation', pdet=True)
      >>> # Calculate SNR first
      >>> snr_result = snroptimal_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
      >>> # Calculate detection probability manually
      >>> pdet_result = snr.pdet(snr_result, snr_th=8.0, type='matched_filter')

      >>> # Using different thresholds for different detectors
      >>> pdet_result = snr.pdet(snr_result, snr_th=[8.0, 8.0, 7.0], type='bool')



      ..
          !! processed by numpydoc !!

   .. py:method:: horizon_distance_analytical(mass_1=1.4, mass_2=1.4, snr_th=None, snr_th_net=None)

      
      Function to calculate detector horizon distance for compact binary coalescences. The horizon distance represents the maximum range at which a source can be detected with optimal orientation and sky location.

      Following Allen et. al. 2013, this method computes the horizon distance for each detector in the network, defined as the luminosity distance at which a compact binary coalescence would produce a signal-to-noise ratio equal to the detection threshold. This assumes optimal orientation with inclination angle theta_jn=0 (face-on) and and the antenna response patterns are at their maximum (overhead), i.e. np.sqrt(F_plus^2 + F_cross^2)=1.

      d_hor = (1/SNR_th) * Partial_SNR = (1/SNR_th) * (SNR_100Mpc * d_eff100Mpc)

      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar masses. Default is 1.4.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar masses. Default is 1.4.

          **snr_th** : `float` or `None`
              SNR threshold for individual detector detection. If None, uses :attr:`~snr_th`. Default is None.

          **snr_th_net** : `float` or `None`
              SNR threshold for network detection. If None, uses :attr:`~snr_th_net`. Default is None.

      :Returns:

          **horizon** : `dict`
              Dictionary containing horizon distances for each detector and network.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'net'.
              Values are horizon distances in Mpc for the given binary system and SNR thresholds.








      .. rubric:: Notes

      - Uses optimal orientation: theta_jn=0 (face-on), ra=dec=psi=phase=0 (overhead)
      - Reference luminosity distance is 100 Mpc for SNR calculation scaling
      - Horizon distance = (d_eff/SNR_th) × SNR_100Mpc where d_eff is effective distance
      - Network horizon uses quadrature sum of effective distances from all detectors
      - Compatible with all waveform approximants supported by the inner product method
      - Uses :meth:`~optimal_snr_with_inner_product` for reference SNR calculation at 100 Mpc


      .. rubric:: Examples

      >>> from gwsnr import GWSNR
      >>> snr = GWSNR(snr_method='inner_product')
      >>> # Calculate BNS horizon for default 1.4+1.4 solar mass system
      >>> horizon = snr.horizon_distance()
      >>> print(f"LIGO-Hanford horizon: {horizon['H1']:.1f} Mpc")

      >>> # Calculate horizon for different mass system
      >>> horizon_bbh = snr.horizon_distance(mass_1=30.0, mass_2=30.0, snr_th=8.0)
      >>> print(f"Network horizon: {horizon_bbh['net']:.1f} Mpc")



      ..
          !! processed by numpydoc !!

   .. py:method:: horizon_distance_numerical(mass_1=1.4, mass_2=1.4, psi=0.0, phase=0.0, geocent_time=1246527224.169434, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, snr_th=None, snr_th_net=None, use_maximization_function=False, minimize_function_dict=None, root_scalar_dict=None, maximization_check=False)

      
      Function to calculate detector horizon distance for compact binary coalescences with optimal sky location.

      Algorithm:
      - For individual detector:
          - For each detector, find the sky location (ra, dec) that maximizes (F_plus^2 + F_cross^2) for a given binary system and geocentric time.
      - For network of detectors:
          - Find the sky location (ra, dec) that maximizes the network SNR (quadrature sum of individual detector SNRs) for the given binary system and geocentric time.
      - For the optimal sky location, compute the horizon distance as the luminosity distance at which the SNR equals the detection threshold.

      :Parameters:

          **mass_1** : `numpy.ndarray` or `float`
              Primary mass of the binary in solar masses. Default is 1.4.

          **mass_2** : `numpy.ndarray` or `float`
              Secondary mass of the binary in solar masses. Default is 1.4.

          **psi** : `numpy.ndarray` or `float`
              Gravitational wave polarization angle in radians. Default is 0.0.

          **phase** : `numpy.ndarray` or `float`
              Gravitational wave phase at coalescence in radians. Default is 0.0.

          **geocent_time** : `float`
              GPS time of coalescence at geocenter in seconds. Default is 1246527224.169434.

          **a_1** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the primary object. Default is 0.0.

          **a_2** : `numpy.ndarray` or `float`
              Dimensionless spin magnitude of the secondary object. Default is 0.0.

          **tilt_1** : `numpy.ndarray` or `float`
              Tilt angle of primary spin relative to orbital angular momentum in radians. Default is 0.0.

          **tilt_2** : `numpy.ndarray` or `float`
              Tilt angle of secondary spin relative to orbital angular momentum in radians. Default is 0. 0.

          **phi_12** : `numpy.ndarray` or `float`
              Azimuthal angle between the two spins in radians. Default is 0.0.

          **phi_jl** : `numpy.ndarray` or `float`
              Azimuthal angle between total and orbital angular momentum in radians. Default is 0.0.

          **lambda_1** : `numpy.ndarray` or `float`
              Dimensionless tidal deformability of the primary object. Default is 0.0.

          **lambda_2** : `numpy.ndarray` or `float`
              Dimensionless tidal deformability of the secondary object. Default is 0.0.

          **eccentricity** : `numpy.ndarray` or `float`
              Orbital eccentricity of the binary system. Default is 0.0.

          **snr_th** : `float` or `None`
              ..

      :Returns:

          **horizon** : `dict`
              Dictionary containing horizon distances for each detector and network.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are horizon distances in Mpc for the given binary system and SNR thresholds.

          **sky_location** : `dict`
              Dictionary containing optimal sky coordinates (for the given geocent_time) for maximum SNR. This is wrt the geocentric time.
              Keys include detector names (e.g., 'L1', 'H1', 'V1') and 'snr_net'.
              Values are tuples (ra, dec) in radians where maximum SNR is achieved.

          **geocent_time** : `float`
              The geocentric time used for the horizon distance calculation, default is 1246527224.169434.













      ..
          !! processed by numpydoc !!


