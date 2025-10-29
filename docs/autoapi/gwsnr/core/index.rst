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




.. py:class:: GWSNR(npool=int(4), snr_method='interpolation_no_spins', snr_type='optimal_snr', gwsnr_verbose=True, multiprocessing_verbose=True, pdet_kwargs=None, mtot_min=2 * 4.98, mtot_max=2 * 112.5 + 10.0, ratio_min=0.1, ratio_max=1.0, spin_max=0.99, mtot_resolution=200, ratio_resolution=20, spin_resolution=10, batch_size_interpolation=1000000, interpolator_dir='./interpolator_pickle', create_new_interpolator=False, sampling_frequency=2048.0, waveform_approximant='IMRPhenomD', frequency_domain_source_model='lal_binary_black_hole', minimum_frequency=20.0, reference_frequency=None, duration_max=None, duration_min=None, fixed_duration=None, mtot_cut=False, psds=None, ifos=None, noise_realization=None, ann_path_dict=None, snr_recalculation=False, snr_recalculation_range=[6, 14], snr_recalculation_waveform_approximant='IMRPhenomXPHM')


   
       Calculate SNR and detection probability for gravitational wave signals from compact binaries.

       Provides multiple computational methods for optimal SNR calculation:
       - Interpolation: Fast calculation using precomputed coefficients
       - Inner product: Direct computation with LAL/Ripple waveforms
       - JAX/MLX: GPU-accelerated computation
       - ANN: Neural network-based estimation

       Other features include:
       - observed SNR based Pdet calculation with various statistical models
       - Horizon distance estimation for detectors and detector networks

   :Parameters:

       **npool** : int, default=4
           Number of processors for parallel computation.

       **mtot_min** : float, default=9.96
           Minimum total mass (solar masses) for interpolation grid.

       **mtot_max** : float, default=235.0
           Maximum total mass (solar masses). Auto-adjusted if mtot_cut=True.

       **ratio_min** : float, default=0.1
           Minimum mass ratio (m2/m1) for interpolation.

       **ratio_max** : float, default=1.0
           Maximum mass ratio for interpolation.

       **spin_max** : float, default=0.99
           Maximum aligned spin magnitude.

       **mtot_resolution** : int, default=200
           Grid points for total mass interpolation.

       **ratio_resolution** : int, default=20
           Grid points for mass ratio interpolation.

       **spin_resolution** : int, default=10
           Grid points for spin interpolation (aligned-spin methods).

       **batch_size_interpolation** : int, default=1000000
           Batch size for interpolation calculations.

       **sampling_frequency** : float, default=2048.0
           Detector sampling frequency (Hz).

       **waveform_approximant** : str, default='IMRPhenomD'
           Waveform model: 'IMRPhenomD', 'IMRPhenomXPHM', 'TaylorF2', etc.

       **frequency_domain_source_model** : str, default='lal_binary_black_hole'
           LAL source model for waveform generation.

       **minimum_frequency** : float, default=20.0
           Minimum frequency (Hz) for waveform generation.

       **reference_frequency** : float, optional
           Reference frequency (Hz). Defaults to minimum_frequency.

       **duration_max** : float, optional
           Maximum waveform duration (seconds). Auto-set for some approximants.

       **duration_min** : float, optional
           Minimum waveform duration (seconds).

       **fixed_duration** : float, optional
           Fixed duration (seconds) for all waveforms.

       **mtot_cut** : bool, default=False
           If True, limit mtot_max based on minimum_frequency.

       **snr_method** : str, default='interpolation_no_spins'
           SNR calculation method. Options:
           - 'interpolation_no_spins[_numba/_jax/_mlx]'
           - 'interpolation_aligned_spins[_numba/_jax/_mlx]'
           - 'inner_product[_jax]'
           - 'ann'

       **snr_type** : str, default='optimal_snr'
           SNR type: 'optimal_snr' or 'observed_snr' (not implemented).

       **noise_realization** : array_like, optional
           Noise realization for observed SNR (not implemented).

       **psds** : dict, optional
           Detector power spectral densities:
           - None: Use bilby defaults
           - {'H1': 'aLIGODesign', 'L1': 'aLIGODesign'}: PSD names
           - {'H1': 'custom_psd.txt'}: Custom PSD files
           - {'H1': 1234567890}: GPS time for data-based PSD

       **ifos** : list, optional
           Custom interferometer objects. Defaults from psds if None.

       **interpolator_dir** : str, default='./interpolator_pickle'
           Directory for storing interpolation coefficients.

       **create_new_interpolator** : bool, default=False
           If True, regenerate interpolation coefficients.

       **gwsnr_verbose** : bool, default=True
           Print initialization parameters.

       **multiprocessing_verbose** : bool, default=True
           Show progress bars during computation.

       **pdet_kwargs** : dict, optional
           Detection probability parameters:
           - 'snr_th': Single detector threshold (default=10.0)
           - 'snr_th_net': Network threshold (default=10.0)
           - 'pdet_type': 'boolean' or 'probability_distribution'
           - 'distribution_type': 'gaussian' or 'noncentral_chi2'

       **ann_path_dict** : dict or str, optional
           Paths to ANN models. Uses built-in models if None.

       **snr_recalculation** : bool, default=False
           Enable hybrid recalculation near detection threshold.

       **snr_recalculation_range** : list, default=[6,14]
           SNR range [min, max] for triggering recalculation.

       **snr_recalculation_waveform_approximant** : str, default='IMRPhenomXPHM'
           Waveform approximant for recalculation.









   .. rubric:: Notes

   - Interpolation methods: fastest for population studies
   - Inner product methods: most accurate for individual events
   - JAX/MLX methods: leverage GPU acceleration
   - ANN methods: fast detection probability, lower SNR accuracy


   .. rubric:: Examples

       Basic interpolation usage:

       >>> from gwsnr import GWSNR
       >>> gwsnr = GWSNR()
       >>> snrs = gwsnr.optimal_snr(mass_1=30, mass_2=30, luminosity_distance=1000, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
       >>> pdet = gwsnr.pdet(mass_1=30, mass_2=30, luminosity_distance=1000, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
       >>> print(f"SNR value: {snrs},
   P_det value: {pdet}")



   ..
       !! processed by numpydoc !!
   .. py:attribute:: npool
      :value: 'None'

      
      ``int``

      Number of processors for parallel processing.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_min
      :value: 'None'

      
      ``float``

      Minimum total mass (Mo) for interpolation grid.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_max
      :value: 'None'

      
      ``float``

      Maximum total mass (Mo) for interpolation grid.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_min
      :value: 'None'

      
      ``float``

      Minimum mass ratio (q = m2/m1) for interpolation grid.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_max
      :value: 'None'

      
      ``float``

      Maximum mass ratio for interpolation grid.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: spin_max
      :value: 'None'

      
      ``float``

      Maximum aligned spin magnitude for interpolation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_resolution
      :value: 'None'

      
      ``int``

      Grid resolution for total mass interpolation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_resolution
      :value: 'None'

      
      ``int``

      Grid resolution for mass ratio interpolation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: spin_resolution
      :value: 'None'

      
      ``int``

      Grid resolution for aligned spin interpolation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ratio_arr
      :value: 'None'

      
      ``numpy.ndarray``

      Mass ratio interpolation grid points.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: mtot_arr
      :value: 'None'

      
      ``numpy.ndarray``

      Total mass interpolation grid points.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: a_1_arr
      :value: 'None'

      
      ``numpy.ndarray``

      Primary aligned spin interpolation grid.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: a_2_arr
      :value: 'None'

      
      ``numpy.ndarray``

      Secondary aligned spin interpolation grid.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: sampling_frequency
      :value: 'None'

      
      ``float``

      Detector sampling frequency (Hz).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: waveform_approximant
      :value: 'None'

      
      ``str``

      LAL waveform approximant (e.g., 'IMRPhenomD', 'IMRPhenomXPHM').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: frequency_domain_source_model
      :value: 'None'

      
      ``str``

      LAL frequency domain source model.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_min
      :value: 'None'

      
      ``float``

      Minimum waveform frequency (Hz).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: f_ref
      :value: 'None'

      
      ``float``

      Reference frequency (Hz) for waveform generation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: duration_max
      :value: 'None'

      
      ``float`` or ``None``

      Maximum waveform duration (s). Auto-set if None.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: duration_min
      :value: 'None'

      
      ``float`` or ``None``

      Minimum waveform duration (s). Auto-set if None.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_method
      :value: 'None'

      
      ``str``

      SNR calculation method. Options: interpolation variants, inner_product variants, ann.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_type
      :value: 'None'

      
      ``str``

      SNR type: 'optimal_snr' or 'observed_snr' (not implemented).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: noise_realization
      :value: 'None'

      
      ``numpy.ndarray`` or ``None``

      Noise realization for observed SNR (not implemented).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: psds_list
      :value: 'None'

      
      ``list`` of ``PowerSpectralDensity``

      Detector power spectral densities.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: detector_tensor_list
      :value: 'None'

      
      ``list`` of ``numpy.ndarray``

      Detector tensors for antenna response calculations.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: detector_list
      :value: 'None'

      
      ``list`` of ``str``

      Detector names (e.g., ['H1', 'L1', 'V1']).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ifos
      :value: 'None'

      
      ``list`` of ``Interferometer``

      Bilby interferometer objects.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: interpolator_dir
      :value: 'None'

      
      ``str``

      Directory for interpolation coefficient storage.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: path_interpolator
      :value: 'None'

      
      ``list`` of ``str``

      Paths to interpolation coefficient files.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_partialsacaled_list
      :value: 'None'

      
      ``list`` of ``numpy.ndarray``

      Partial-scaled SNR interpolation coefficients.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: multiprocessing_verbose
      :value: 'None'

      
      ``bool``

      Show progress bars for multiprocessing computations.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: param_dict_given
      :value: 'None'

      
      ``dict``

      Interpolator parameter dictionary for caching.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_th
      :value: 'None'

      
      ``float``

      Individual detector SNR threshold (default: 8.0).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_th_net
      :value: 'None'

      
      ``float``

      Network SNR threshold (default: 8.0).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: model_dict
      :value: 'None'

      
      ``dict``

      ANN models for each detector (when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: scaler_dict
      :value: 'None'

      
      ``dict``

      ANN feature scalers for each detector (when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: error_adjustment
      :value: 'None'

      
      ``dict``

      ANN error correction parameters (when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: ann_catalogue
      :value: 'None'

      
      ``dict``

      ANN model configuration and paths (when snr_method='ann').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_recalculation
      :value: 'None'

      
      ``bool``

      Enable hybrid SNR recalculation near detection threshold.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_recalculation_range
      :value: 'None'

      
      ``list``

      SNR range [min, max] triggering recalculation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: snr_recalculation_waveform_approximant
      :value: 'None'

      
      ``str``

      Waveform approximant for SNR recalculation.















      ..
          !! processed by numpydoc !!

   .. py:attribute:: get_interpolated_snr
      :value: 'None'

      
      ``function``

      Interpolated SNR calculation function (backend-specific).















      ..
          !! processed by numpydoc !!

   .. py:attribute:: noise_weighted_inner_product_jax
      :value: 'None'

      
      ``function``

      JAX-accelerated inner product function (when snr_method='inner_product_jax').















      ..
          !! processed by numpydoc !!

   .. py:attribute:: pdet_kwargs
      :value: 'None'

      

   .. py:attribute:: fixed_duration
      :value: 'None'

      

   .. py:attribute:: batch_size_interpolation
      :value: '1000000'

      

   .. py:attribute:: mtot_cut
      :value: 'False'

      

   .. py:method:: interpolator_setup(interpolator_dir, create_new_interpolator, psds_list, detector_tensor_list, detector_list)

      
      Set up interpolator files for fast SNR calculation using precomputed coefficients.

      This method manages the creation and loading of partialscaled SNR interpolation data.
      It checks for existing interpolators, generates missing ones, and loads coefficients
      for runtime use.

      :Parameters:

          **interpolator_dir** : str
              Directory path for storing interpolator pickle files.

          **create_new_interpolator** : bool
              If True, generates new interpolators regardless of existing files.

          **psds_list** : list
              Power spectral density objects for each detector.

          **detector_tensor_list** : list
              Detector tensor arrays for antenna response calculations.

          **detector_list** : list
              Detector names (e.g., ['L1', 'H1', 'V1']).

      :Returns:

          **path_interpolator_all** : list
              File paths to interpolator pickle files for all detectors.








      .. rubric:: Notes

      - Uses :func:`interpolator_check` to identify missing interpolators
      - Calls :meth:`init_partialscaled` to generate new coefficients
      - Loads coefficients into :attr:`snr_partialsacaled_list` for runtime use





      ..
          !! processed by numpydoc !!

   .. py:method:: ann_initilization(ann_path_dict, detector_list, sampling_frequency, minimum_frequency, waveform_approximant)

      
      Initialize ANN models and feature scalers for detection probability estimation.

      Loads pre-trained neural network models, feature scalers, and error correction parameters
      for each detector. Validates that model parameters match current GWSNR configuration.

      :Parameters:

          **ann_path_dict** : dict, str, or None
              Dictionary or JSON file path containing ANN model paths for each detector.
              If None, uses default models from gwsnr/ann/data/ann_path_dict.json.
              Expected structure: {detector_name: {'model_path': str, 'scaler_path': str,
              'error_adjustment_path': str, 'sampling_frequency': float, 'minimum_frequency': float,
              'waveform_approximant': str, 'snr_th': float}}.

          **detector_list** : list of str
              Detector names requiring ANN models (e.g., ['L1', 'H1', 'V1']).

          **sampling_frequency** : float
              Sampling frequency in Hz. Must match ANN training configuration.

          **minimum_frequency** : float
              Minimum frequency in Hz. Must match ANN training configuration.

          **waveform_approximant** : str
              Waveform model. Must match ANN training configuration.

          **snr_th** : float
              Detection threshold. Must match ANN training configuration.

      :Returns:

          **model_dict** : dict
              Loaded TensorFlow/Keras models {detector_name: model}.

          **scaler_dict** : dict
              Feature preprocessing scalers {detector_name: scaler}.

          **error_adjustment** : dict
              Post-prediction correction parameters {detector_name: {'slope': float, 'intercept': float}}.

          **ann_catalogue** : dict
              Complete ANN configuration and paths for all detectors.




      :Raises:

          ValueError
              If model not available for detector, or if model parameters don't match
              current GWSNR configuration.




      .. rubric:: Notes

      - Loads models from gwsnr/ann/data if file paths don't exist locally
      - Validates parameter compatibility before loading
      - Error adjustment improves prediction accuracy via linear correction





      ..
          !! processed by numpydoc !!

   .. py:method:: calculate_mtot_max(mtot_max, minimum_frequency)

      
      Calculate maximum total mass cutoff based on minimum frequency to ensure positive chirp time.

      This method finds the maximum total mass where the chirp time becomes zero at the given
      minimum frequency. Systems with higher masses would have negative chirp times, causing
      waveform generation failures. A safety factor of 1.1 is applied.

      :Parameters:

          **mtot_max** : float
              User-specified maximum total mass in solar masses.

          **minimum_frequency** : float
              Minimum frequency in Hz for waveform generation.

      :Returns:

          float
              Adjusted maximum total mass (≤ input mtot_max) ensuring positive chirp time.








      .. rubric:: Notes

      Uses equal mass ratio (q=1.0) as conservative estimate since it maximizes chirp time
      for given total mass. Particularly important for TaylorF2 approximant.





      ..
          !! processed by numpydoc !!

   .. py:method:: print_all_params(verbose=True)

      
      Print all parameters and configuration of the GWSNR class instance.

      Displays computational settings, waveform configuration, detector setup, mass parameter
      ranges, and interpolation parameters for verification and debugging.

      :Parameters:

          **verbose** : bool, default=True
              If True, print all parameters to stdout. If False, suppress output.









      .. rubric:: Notes

      Printed information includes:
      - Computational: processors, SNR method
      - Waveform: approximant, frequencies, sampling rate
      - Detectors: names and PSDs
      - Mass ranges: total mass bounds with frequency cutoffs
      - Interpolation: grid resolutions and bounds (when applicable)

      Called automatically during initialization when gwsnr_verbose=True.





      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr(mass_1=np.array([10.0]), mass_2=np.array([10.0]), luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Calculate optimal SNR for gravitational wave signals from compact binary coalescences.

      This is the primary interface for SNR calculation, routing to the appropriate computational method
      based on the configured snr_method. Supports interpolation, inner product, JAX-accelerated, and
      neural network methods.

      :Parameters:

          **mass_1** : array_like or float, default=np.array([10.0])
              Primary mass in solar masses.

          **mass_2** : array_like or float, default=np.array([10.0])
              Secondary mass in solar masses.

          **luminosity_distance** : array_like or float, default=100.0
              Luminosity distance in Mpc.

          **theta_jn** : array_like or float, default=0.0
              Inclination angle (total angular momentum to line of sight) in radians.

          **psi** : array_like or float, default=0.0
              Polarization angle in radians.

          **phase** : array_like or float, default=0.0
              Coalescence phase in radians.

          **geocent_time** : array_like or float, default=1246527224.169434
              GPS coalescence time at geocenter in seconds.

          **ra** : array_like or float, default=0.0
              Right ascension in radians.

          **dec** : array_like or float, default=0.0
              Declination in radians.

          **a_1** : array_like or float, default=0.0
              Primary spin magnitude (dimensionless).

          **a_2** : array_like or float, default=0.0
              Secondary spin magnitude (dimensionless).

          **tilt_1** : array_like or float, default=0.0
              Primary spin tilt angle in radians.

          **tilt_2** : array_like or float, default=0.0
              Secondary spin tilt angle in radians.

          **phi_12** : array_like or float, default=0.0
              Azimuthal angle between spins in radians.

          **phi_jl** : array_like or float, default=0.0
              Azimuthal angle between total and orbital angular momentum in radians.

          **lambda_1** : array_like or float, default=0.0
              Primary tidal deformability (dimensionless).

          **lambda_2** : array_like or float, default=0.0
              Secondary tidal deformability (dimensionless).

          **eccentricity** : array_like or float, default=0.0
              Orbital eccentricity at reference frequency.

          **gw_param_dict** : dict or bool, default=False
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are detector names
              ('H1', 'L1', 'V1', etc.) and 'snr_net'. Values are arrays matching input size.








      .. rubric:: Notes

      - For interpolation methods, tilt angles are converted to aligned spins: a_i * cos(tilt_i)
      - Total mass must be within [mtot_min, mtot_max] for non-zero SNR
      - Hybrid recalculation uses higher-order waveforms near detection threshold if enabled
      - Compatible with all configured detector networks and waveform approximants


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='interpolation_no_spins')
      >>> result = snr.optimal_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
      >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")

      >>> # Multiple systems with parameter dictionary
      >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
      >>> result = snr.optimal_snr(gw_param_dict=params)



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_ann(mass_1=30.0, mass_2=29.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Calculate SNR using artificial neural network (ANN) prediction.

      Uses pre-trained neural networks to rapidly estimate optimal SNR for gravitational wave
      signals with arbitrary spin configurations. The method first computes partial-scaled SNR
      via interpolation, then feeds this along with other intrinsic parameters to detector-specific
      ANN models for fast SNR prediction.

      :Parameters:

          **mass_1** : array_like or float, default=30.0
              Primary mass in solar masses.

          **mass_2** : array_like or float, default=29.0
              Secondary mass in solar masses.

          **luminosity_distance** : array_like or float, default=100.0
              Luminosity distance in Mpc.

          **theta_jn** : array_like or float, default=0.0
              Inclination angle in radians.

          **psi** : array_like or float, default=0.0
              Polarization angle in radians.

          **phase** : array_like or float, default=0.0
              Coalescence phase in radians.

          **geocent_time** : array_like or float, default=1246527224.169434
              GPS coalescence time at geocenter in seconds.

          **ra** : array_like or float, default=0.0
              Right ascension in radians.

          **dec** : array_like or float, default=0.0
              Declination in radians.

          **a_1** : array_like or float, default=0.0
              Primary spin magnitude (dimensionless).

          **a_2** : array_like or float, default=0.0
              Secondary spin magnitude (dimensionless).

          **tilt_1** : array_like or float, default=0.0
              Primary tilt angle in radians.

          **tilt_2** : array_like or float, default=0.0
              Secondary tilt angle in radians.

          **phi_12** : array_like or float, default=0.0
              Azimuthal angle between spins in radians.

          **phi_jl** : array_like or float, default=0.0
              Azimuthal angle between total and orbital angular momentum in radians.

          **gw_param_dict** : dict or bool, default=False
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR estimates for each detector and network. Keys are detector names
              ('H1', 'L1', 'V1', etc.) and 'snr_net'.








      .. rubric:: Notes

      - Requires pre-trained ANN models loaded during initialization
      - Uses aligned spin components: a_i * cos(tilt_i) for effective spin calculation
      - ANN inputs: partial-scaled SNR, amplitude factor, mass ratio, effective spin, inclination
      - Applies error correction to improve prediction accuracy
      - Total mass must be within [mtot_min, mtot_max] for valid results


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='ann')
      >>> result = snr.optimal_snr_with_ann(mass_1=30, mass_2=25, a_1=0.5, tilt_1=0.2)
      >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: output_ann(idx, params)

      
      Prepare ANN input features from gravitational wave parameters.

      Transforms gravitational wave parameters into feature vectors for neural network
      prediction. Calculates partial-scaled SNR via interpolation and combines with
      intrinsic parameters to create standardized input features.

      :Parameters:

          **idx** : numpy.ndarray of bool
              Boolean mask for valid mass ranges (mtot_min <= mtot <= mtot_max).

          **params** : dict
              GW parameter dictionary with keys: mass_1, mass_2, luminosity_distance,
              theta_jn, a_1, a_2, tilt_1, tilt_2, psi, geocent_time, ra, dec.

      :Returns:

          list of numpy.ndarray
              Feature arrays for each detector, shape (N, 5) with columns:
              [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn].








      .. rubric:: Notes

      - Uses aligned spin components: a_i * cos(tilt_i)
      - Amplitude factor: A1 = Mc^(5/6) / d_eff
      - Effective spin: chi_eff = (m1*a1z + m2*a2z) / (m1+m2)





      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_interpolation(mass_1=30.0, mass_2=29.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, output_jsonfile=False, gw_param_dict=False)

      
      Calculate SNR (for non-spinning or aligned-spin) using bicubic interpolation of precomputed coefficients.

      Fast SNR calculation method using interpolated partial-scaled SNR values across
      intrinsic parameter grids. Supports no-spin and aligned-spin configurations with
      Numba or JAX acceleration for population studies.

      :Parameters:

          **mass_1** : array_like or float, default=30.0
              Primary mass in solar masses.

          **mass_2** : array_like or float, default=29.0
              Secondary mass in solar masses.

          **luminosity_distance** : array_like or float, default=100.0
              Luminosity distance in Mpc.

          **theta_jn** : array_like or float, default=0.0
              Inclination angle in radians.

          **psi** : array_like or float, default=0.0
              Polarization angle in radians.

          **phase** : array_like or float, default=0.0
              Coalescence phase in radians.

          **geocent_time** : array_like or float, default=1246527224.169434
              GPS coalescence time at geocenter in seconds.

          **ra** : array_like or float, default=0.0
              Right ascension in radians.

          **dec** : array_like or float, default=0.0
              Declination in radians.

          **a_1** : array_like or float, default=0.0
              Primary aligned spin component (for aligned-spin methods only).

          **a_2** : array_like or float, default=0.0
              Secondary aligned spin component (for aligned-spin methods only).

          **gw_param_dict** : dict or bool, default=False
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are detector names
              ('H1', 'L1', 'V1', etc.) and 'snr_net'. Systems outside mass bounds have zero SNR.








      .. rubric:: Notes

      - Requires precomputed interpolation coefficients from class initialization
      - self.get_interpolated_snr is set based on snr_method (Numba or JAX or MLX) and whether the system is non-spinning or aligned-spin
      - Total mass must be within [mtot_min, mtot_max] for valid results
      - Uses aligned spin: a_i * cos(tilt_i) for spin-enabled methods
      - Backend acceleration available via JAX or Numba depending on snr_method


      .. rubric:: Examples

      >>> snr_calc = GWSNR(snr_method='interpolation_no_spins')
      >>> result = snr_calc.optimal_snr_with_interpolation(mass_1=30, mass_2=25)
      >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: init_partialscaled()

      
      Generate partial-scaled SNR interpolation coefficients for fast bicubic interpolation.

      Computes and saves distance-independent SNR coefficients across intrinsic parameter grids
      for each detector. These coefficients enable fast runtime SNR calculation via interpolation
      without requiring waveform generation.

      Creates parameter grids based on interpolation method:
      - No-spin: 2D grid (mass_ratio, total_mass)
      - Aligned-spin: 4D grid (mass_ratio, total_mass, a_1, a_2)

      For each grid point, computes optimal SNR with fixed extrinsic parameters
      (d_L=100 Mpc, θ_jn=0, overhead sky location), then scales by effective distance
      and chirp mass: partial_SNR = (optimal_SNR × d_eff) / Mc^(5/6).

      Coefficients are saved as pickle files for runtime interpolation.






      :Raises:

          ValueError
              If mtot_min < 1.0 or snr_method not supported for interpolation.




      .. rubric:: Notes

      Grid dimensions set by ratio_resolution, mtot_resolution, spin_resolution.
      Automatically called during initialization when coefficients missing.





      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_inner_product(mass_1=10, mass_2=10, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Calculate optimal SNR using LAL waveform generation and noise-weighted inner products.

      This method computes SNR by generating gravitational wave signals with LAL and calculating
      matched filtering inner products against detector noise PSDs. Supports all LAL waveform
      approximants including aligned and precessing spin systems.

      :Parameters:

          **mass_1** : array_like or float, default=10
              Primary mass in solar masses.

          **mass_2** : array_like or float, default=10
              Secondary mass in solar masses.

          **luminosity_distance** : array_like or float, default=100.0
              Luminosity distance in Mpc.

          **theta_jn** : array_like or float, default=0.0
              Inclination angle in radians.

          **psi** : array_like or float, default=0.0
              Polarization angle in radians.

          **phase** : array_like or float, default=0.0
              Coalescence phase in radians.

          **geocent_time** : array_like or float, default=1246527224.169434
              GPS coalescence time at geocenter in seconds.

          **ra** : array_like or float, default=0.0
              Right ascension in radians.

          **dec** : array_like or float, default=0.0
              Declination in radians.

          **a_1** : array_like or float, default=0.0
              Primary spin magnitude (dimensionless).

          **a_2** : array_like or float, default=0.0
              Secondary spin magnitude (dimensionless).

          **tilt_1** : array_like or float, default=0.0
              Primary spin tilt angle in radians.

          **tilt_2** : array_like or float, default=0.0
              Secondary spin tilt angle in radians.

          **phi_12** : array_like or float, default=0.0
              Azimuthal angle between spins in radians.

          **phi_jl** : array_like or float, default=0.0
              Azimuthal angle between total and orbital angular momentum in radians.

          **lambda_1** : array_like or float, default=0.0
              Primary tidal deformability (dimensionless).

          **lambda_2** : array_like or float, default=0.0
              Secondary tidal deformability (dimensionless).

          **eccentricity** : array_like or float, default=0.0
              Orbital eccentricity at reference frequency.

          **gw_param_dict** : dict or bool, default=False
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are detector names
              ('H1', 'L1', 'V1', etc.) and 'snr_net'. Systems outside mass bounds have zero SNR.








      .. rubric:: Notes

      - Waveform duration auto-estimated from chirp time with 1.1x safety factor
      - Uses multiprocessing for parallel computation across npool processors
      - Requires 'if __name__ == "__main__":' guard when using multiprocessing
      - Most accurate method but slower than interpolation for population studies


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product')
      >>> result = snr.optimal_snr_with_inner_product(mass_1=30, mass_2=25)
      >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_inner_product_ripple(mass_1=10, mass_2=10, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=False, output_jsonfile=False)

      
      Calculate optimal SNR using JAX-accelerated Ripple waveforms and noise-weighted inner products.

      Uses the Ripple waveform generator with JAX backend for fast SNR computation via
      vectorized inner products. Supports arbitrary spin configurations and provides
      significant speedup over LAL-based methods for population studies.

      :Parameters:

          **mass_1** : array_like or float, default=10
              Primary mass in solar masses.

          **mass_2** : array_like or float, default=10
              Secondary mass in solar masses.

          **luminosity_distance** : array_like or float, default=100.0
              Luminosity distance in Mpc.

          **theta_jn** : array_like or float, default=0.0
              Inclination angle in radians.

          **psi** : array_like or float, default=0.0
              Polarization angle in radians.

          **phase** : array_like or float, default=0.0
              Coalescence phase in radians.

          **geocent_time** : array_like or float, default=1246527224.169434
              GPS coalescence time at geocenter in seconds.

          **ra** : array_like or float, default=0.0
              Right ascension in radians.

          **dec** : array_like or float, default=0.0
              Declination in radians.

          **a_1** : array_like or float, default=0.0
              Primary spin magnitude (dimensionless).

          **a_2** : array_like or float, default=0.0
              Secondary spin magnitude (dimensionless).

          **tilt_1** : array_like or float, default=0.0
              Primary spin tilt angle in radians.

          **tilt_2** : array_like or float, default=0.0
              Secondary spin tilt angle in radians.

          **phi_12** : array_like or float, default=0.0
              Azimuthal angle between spins in radians.

          **phi_jl** : array_like or float, default=0.0
              Azimuthal angle between total and orbital angular momentum in radians.

          **lambda_1** : array_like or float, default=0.0
              Primary tidal deformability (dimensionless).

          **lambda_2** : array_like or float, default=0.0
              Secondary tidal deformability (dimensionless).

          **eccentricity** : array_like or float, default=0.0
              Orbital eccentricity at reference frequency.

          **gw_param_dict** : dict or bool, default=False
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are detector names
              ('H1', 'L1', 'V1', etc.) and 'snr_net'. Systems outside mass bounds have zero SNR.








      .. rubric:: Notes

      - Requires snr_method='inner_product_jax' during initialization
      - Uses JAX JIT compilation and vectorization for GPU acceleration
      - Duration auto-estimated with safety bounds from duration_min/max
      - Compatible with Ripple-supported approximants (IMRPhenomD, IMRPhenomXPHM)
      - Supports precessing spins through full parameter space


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product_jax')
      >>> result = snr.optimal_snr_with_inner_product_ripple(mass_1=30, mass_2=25)
      >>> print(f"Network SNR: {result['snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: pdet(mass_1=np.array([10.0]), mass_2=np.array([10.0]), luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=False, output_jsonfile=False, snr_th=None, snr_th_net=None, pdet_type=None, distribution_type=None, include_optimal_snr=False, include_observed_snr=False)

      
      Calculate probability of detection for gravitational wave signals.

      Computes detection probability based on SNR thresholds for individual detectors and detector networks. Accounts for noise fluctuations by modeling observed SNR as statistical distributions around optimal SNR values.

      :Parameters:

          **mass_1** : array_like or float, default=np.array([10.0])
              Primary mass in solar masses.

          **mass_2** : array_like or float, default=np.array([10.0])
              Secondary mass in solar masses.

          **luminosity_distance** : array_like or float, default=100.0
              Luminosity distance in Mpc.

          **theta_jn** : array_like or float, default=0.0
              Inclination angle in radians.

          **psi** : array_like or float, default=0.0
              Polarization angle in radians.

          **phase** : array_like or float, default=0.0
              Coalescence phase in radians.

          **geocent_time** : array_like or float, default=1246527224.169434
              GPS coalescence time at geocenter in seconds.

          **ra** : array_like or float, default=0.0
              Right ascension in radians.

          **dec** : array_like or float, default=0.0
              Declination in radians.

          **a_1** : array_like or float, default=0.0
              Primary spin magnitude (dimensionless).

          **a_2** : array_like or float, default=0.0
              Secondary spin magnitude (dimensionless).

          **tilt_1** : array_like or float, default=0.0
              Primary spin tilt angle in radians.

          **tilt_2** : array_like or float, default=0.0
              Secondary spin tilt angle in radians.

          **phi_12** : array_like or float, default=0.0
              Azimuthal angle between spins in radians.

          **phi_jl** : array_like or float, default=0.0
              Azimuthal angle between total and orbital angular momentum in radians.

          **lambda_1** : array_like or float, default=0.0
              Primary tidal deformability (dimensionless).

          **lambda_2** : array_like or float, default=0.0
              Secondary tidal deformability (dimensionless).

          **eccentricity** : array_like or float, default=0.0
              Orbital eccentricity at reference frequency.

          **gw_param_dict** : dict or bool, default=False
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'pdet.json'.

          **snr_th** : float, array_like, or None, default=None
              SNR threshold for individual detectors. If None, uses pdet_kwargs['snr_th'].
              If array, must match number of detectors.

          **snr_th_net** : float or None, default=None
              Network SNR threshold. If None, uses pdet_kwargs['snr_th_net'].

          **pdet_type** : str or None, default=None
              Detection probability method:
              - 'boolean': Binary detection (0 or 1) based on noise realizations
              - 'probability_distribution': Analytical probability using noise statistics
              If None, uses pdet_kwargs['pdet_type'].

          **distribution_type** : str or None, default=None
              Noise model for observed SNR:
              - 'gaussian': Gaussian noise (sigma=1)
              - 'noncentral_chi2': Non-central chi-squared (2 DOF per detector)
              If None, uses pdet_kwargs['distribution_type'].
              - 'fixed_snr': Deterministic detection based on optimal SNR (only for 'boolean' pdet_type)

      :Returns:

          dict
              Detection probabilities for each detector and network. Keys are detector
              names ('H1', 'L1', 'V1', etc.) and 'pdet_net'. Values depend on pdet_type:
              - 'boolean': Binary arrays (0/1) indicating detection
              - 'probability_distribution': Probability arrays (0-1)








      .. rubric:: Notes

      - First computes optimal SNR using configured snr_method
      - Models observed SNR as noisy version of optimal SNR
      - Non-central chi-squared uses 2 DOF per detector, network uses 2×N_det DOF
      - Boolean method generates random noise realizations for each system
      - Probability method uses analytical CDFs for faster computation


      .. rubric:: Examples

      >>> pdet_calc = GWSNR(pdet_kwargs={'snr_th': 8, 'pdet_type': 'boolean'})
      >>> result = pdet_calc.pdet(mass_1=30, mass_2=25, luminosity_distance=200)
      >>> print(f"Network detection: {result['pdet_net'][0]}")

      >>> # Analytical probability calculation
      >>> pdet_calc = GWSNR(pdet_kwargs={'pdet_type': 'probability_distribution'})
      >>> probs = pdet_calc.pdet(mass_1=[20,30], mass_2=[20,25], luminosity_distance=150)



      ..
          !! processed by numpydoc !!

   .. py:method:: horizon_distance_analytical(mass_1=1.4, mass_2=1.4, snr_th=None)

      
      Calculate detector horizon distance for compact binary coalescences. Follows analytical formula from arXiv:gr-qc/0509116 .

      This method doesn't calculate horizon distance for the detector network, but for individual detectors only. Use horizon_distance_numerical for network horizon.

      Computes the maximum range at which a source can be detected with optimal orientation (face-on, overhead). Uses reference SNR at 100 Mpc scaled by  effective distance and detection threshold.

      :Parameters:

          **mass_1** : array_like or float, default=1.4
              Primary mass in solar masses.

          **mass_2** : array_like or float, default=1.4
              Secondary mass in solar masses.

          **snr_th** : float, optional
              Individual detector SNR threshold. Uses class default if None.

          **snr_th_net** : float, optional
              Network SNR threshold. Uses class default if None.

      :Returns:

          **horizon_distance_dict** : dict
              Horizon distances in Mpc for each detector and network.
              Keys: detector names ('H1', 'L1', etc.) and 'snr_net'.
              Values: array of horizon distances in Mpc.








      .. rubric:: Notes

      - Assumes optimal orientation: θ_jn=0, overhead sky location
      - Formula: d_horizon = (d_eff/SNR_th) x SNR_100Mpc
      - Network horizon uses quadrature sum of detector responses
      - Compatible with all waveform approximants


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product')
      >>> horizon = snr.horizon_distance_analytical(mass_1=1.4, mass_2=1.4)
      >>> print(f"H1 horizon: {horizon['H1']:.1f} Mpc")



      ..
          !! processed by numpydoc !!

   .. py:method:: horizon_distance_numerical(mass_1=1.4, mass_2=1.4, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, snr_th=None, snr_th_net=None, detector_location_as_optimal_sky=False, minimize_function_dict=None, root_scalar_dict=None, maximization_check=False)

      
      Calculate detector horizon distance with optimal sky positioning and arbitrary spin parameters.

      Finds the maximum luminosity distance at which a gravitational wave signal can be
      detected above threshold SNR. For each detector, determines optimal sky location
      that maximizes antenna response, then solves for distance where SNR equals threshold.

      :Parameters:

          **mass_1** : float, default=1.4
              Primary mass in solar masses.

          **mass_2** : float, default=1.4
              Secondary mass in solar masses.

          **psi** : float, default=0.0
              Polarization angle in radians.

          **phase** : float, default=0.0
              Coalescence phase in radians.

          **geocent_time** : float, default=1246527224.169434
              GPS coalescence time at geocenter in seconds.

          **a_1** : float, default=0.0
              Primary spin magnitude (dimensionless).

          **a_2** : float, default=0.0
              Secondary spin magnitude (dimensionless).

          **tilt_1** : float, default=0.0
              Primary spin tilt angle in radians.

          **tilt_2** : float, default=0.0
              Secondary spin tilt angle in radians.

          **phi_12** : float, default=0.0
              Azimuthal angle between spins in radians.

          **phi_jl** : float, default=0.0
              Azimuthal angle between total and orbital angular momentum in radians.

          **lambda_1** : float, default=0.0
              Primary tidal deformability (dimensionless).

          **lambda_2** : float, default=0.0
              Secondary tidal deformability (dimensionless).

          **eccentricity** : float, default=0.0
              Orbital eccentricity at reference frequency.

          **gw_param_dict** : dict or bool, default=False
              Parameter dictionary. If provided, overrides individual arguments.

          **snr_th** : float, optional
              Individual detector SNR threshold. Uses class default if None.

          **snr_th_net** : float, optional
              Network SNR threshold. Uses class default if None.

          **detector_location_as_optimal_sky** : bool, default=False
              If True, uses detector zenith as optimal sky location instead of optimization.

          **minimize_function_dict** : dict, optional
              Parameters for sky location optimization. It contains input for scipy's differential_evolution.
              Default: dict(
                  bounds=[(0, 2*np.pi), (-np.pi/2, np.pi/2)], # ra, dec bounds
                  tol=1e-7,
                  polish=True,
                  maxiter=10000
              )

          **root_scalar_dict** : dict, optional
              Parameters for horizon distance root finding. It contains input for scipy's root_scalar.
              Default: dict(
                  bracket=[1, 100000], # redshift range
                  method='bisect',
                  xtol=1e-5
              )

          **maximization_check** : bool, default=False
              Verify that antenna response maximization achieved ~1.0.

      :Returns:

          **horizon** : dict
              Horizon distances in Mpc for each detector and network ('snr_net').

          **optimal_sky_location** : dict
              Optimal sky coordinates (ra, dec) in radians for maximum SNR at given geocent_time.








      .. rubric:: Notes

      - Uses differential evolution to find optimal sky location maximizing antenna response
      - Network horizon maximizes quadrature sum of detector SNRs
      - Individual detector horizons maximize (F_plus² + F_cross²)
      - Root finding determines distance where SNR equals threshold
      - Computation time depends on optimization tolerances and system complexity


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product')
      >>> horizon, sky = snr.horizon_distance_numerical(mass_1=1.4, mass_2=1.4)
      >>> print(f"Network horizon: {horizon['snr_net']:.1f} Mpc at (RA={sky['snr_net'][0]:.2f}, Dec={sky['snr_net'][1]:.2f})")



      ..
          !! processed by numpydoc !!


