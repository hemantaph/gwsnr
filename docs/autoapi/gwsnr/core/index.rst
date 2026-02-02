:py:mod:`gwsnr.core`
====================

.. py:module:: gwsnr.core


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   core_data/index.rst


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




.. py:class:: GWSNR(npool=int(4), snr_method='interpolation_aligned_spins', snr_type='optimal_snr', gwsnr_verbose=True, multiprocessing_verbose=True, pdet_kwargs=None, mtot_min=2 * 4.98, mtot_max=2 * 112.5 + 10.0, ratio_min=0.1, ratio_max=1.0, spin_max=0.99, mtot_resolution=200, ratio_resolution=20, spin_resolution=10, batch_size_interpolation=1000000, interpolator_dir='./interpolator_json', create_new_interpolator=False, sampling_frequency=2048.0, waveform_approximant='IMRPhenomD', frequency_domain_source_model='lal_binary_black_hole', minimum_frequency=20.0, reference_frequency=None, duration_max=None, duration_min=None, fixed_duration=None, mtot_cut=False, psds=None, ifos=None, noise_realization=None, ann_path_dict=None, snr_recalculation=False, snr_recalculation_range=[6, 14], snr_recalculation_waveform_approximant='IMRPhenomXPHM')


   
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

       **npool** : `int`
           Number of processors for parallel computation.

           default: 4

       **mtot_min** : `float`
           Minimum total mass (solar masses) for interpolation grid.

           default: 9.96

       **mtot_max** : `float`
           Maximum total mass (solar masses). Auto-adjusted if mtot_cut=True.

           default: 235.0

       **ratio_min** : `float`
           Minimum mass ratio (m2/m1) for interpolation.

           default: 0.1

       **ratio_max** : `float`
           Maximum mass ratio for interpolation.

           default: 1.0

       **spin_max** : `float`
           Maximum aligned spin magnitude.

           default: 0.99

       **mtot_resolution** : `int`
           Grid points for total mass interpolation.

           default: 200

       **ratio_resolution** : `int`
           Grid points for mass ratio interpolation.

           default: 20

       **spin_resolution** : `int`
           Grid points for spin interpolation (aligned-spin methods).

           default: 10

       **batch_size_interpolation** : `int`
           Batch size for interpolation calculations.

           default: 1000000

       **sampling_frequency** : `float`
           Detector sampling frequency (Hz).

           default: 2048.0

       **waveform_approximant** : `str`
           Bilby waveform model: 'IMRPhenomD', 'IMRPhenomXPHM', 'TaylorF2', etc.

           default: 'IMRPhenomD'

       **frequency_domain_source_model** : `str`
           Bilby frequency domain source model function for waveform generation.

           default: 'lal_binary_black_hole'

       **minimum_frequency** : `float`
           Minimum frequency (Hz) for waveform generation.

           default: 20.0

       **reference_frequency** : `float`
           Reference frequency (Hz). Optional.

           default: minimum_frequency.

       **duration_max** : `float`
           Maximum waveform duration (seconds). Optional. Auto-set for some approximants.

       **duration_min** : `float`
           Minimum waveform duration (seconds). Optional.

       **fixed_duration** : `float`
           Fixed duration (seconds) for all waveforms. Optional.

       **mtot_cut** : `bool`
           If True, limit mtot_max based on minimum_frequency.

           default: False

       **snr_method** : `str`
           SNR calculation method. Options:

           - 'interpolation_no_spins[_numba/_jax/_mlx]'

           - 'interpolation_aligned_spins[_numba/_jax/_mlx]'

           - 'inner_product[_jax]'

           - 'ann'

           default : 'interpolation_aligned_spins'

       **snr_type** : `str`
           SNR type: 'optimal_snr' or 'observed_snr' (not implemented).

           default: 'optimal_snr'

       **noise_realization** : `numpy.ndarray`
           Noise realization for observed SNR (not implemented). Optional.

       **psds** : `dict`
           Detector power spectral densities. Optional.

           Options:

           - None: Use bilby defaults

           - {'H1': 'aLIGODesign', 'L1': 'aLIGODesign'}: Use bilby default PSD names

           - {'H1': 'custom_psd.txt'} or {'H1': 'custom_asd.txt'}: Use custom PSD/ASD files. File should contain two columns: frequency and PSD/ASD values.

           - {'H1': 1234567890}: Use GPS time for data-based PSD

       **ifos** : `list`
           Custom interferometer objects. Defaults from psds if None. Optional.

           Options:

           - None: Use bilby defaults

           - ['H1', 'L1']: Uses bilby default detector configuration

           - Custom ifos and psds example:
           >>> import bilby
           >>> from gwsnr import GWSNR
           >>> ifosLIO = bilby.gw.detector.interferometer.Interferometer(
                   name = 'LIO',
                   power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file='your_asd.txt'),
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
           Directory for storing interpolation coefficients. Optional.

       **create_new_interpolator** : `bool`
           If True, regenerate interpolation coefficients. Optional.

       **gwsnr_verbose** : `bool`
           Print initialization parameters.

       **multiprocessing_verbose** : `bool`
           Show progress bars during computation.

           default: True

       **pdet_kwargs** : `dict`
           Detection probability parameters.
           Default: {'snr_th': 10.0, 'snr_th_net': 10.0, 'pdet_type': 'boolean', 'distribution_type': 'gaussian'}

       **ann_path_dict** : `dict` or `str`
           Paths to ANN models. Uses built-in models if None. Optional.

       **snr_recalculation** : `bool`
           Enable hybrid recalculation near detection threshold. Optional.

       **snr_recalculation_range** : `list`, default=[6,14]
           SNR range [min, max] for triggering recalculation. Optional.

       **snr_recalculation_waveform_approximant** : `str`
           Waveform approximant for recalculation. Optional.
           Default: 'IMRcd gwsnrPhenomXPHM'









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
   >>> print(f"SNR value: {snrs}")
   >>> print(f"P_det value: {pdet}")

   Instance Methods
   ----------
   GWSNR class has the following methods:

   +------------------------------------------------+------------------------------------------------+
   | Method                                         | Description                                    |
   +================================================+================================================+
   | :meth:`~calculate_mtot_max`                    | Calculate maximum total mass cutoff based on   |
   |                                                | minimum frequency                              |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~optimal_snr`                           | Primary interface for SNR calculation          |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~optimal_snr_with_ann`                  | Calculate SNR using artificial neural network  |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~optimal_snr_with_interpolation`        | Calculate SNR using bicubic interpolation      |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~optimal_snr_with_inner_product`        | Calculate SNR using LAL waveforms and inner    |
   |                                                | products                                       |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~optimal_snr_with_inner_product_ripple` | Calculate SNR using JAX-accelerated Ripple     |
   |                                                | waveforms                                      |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~pdet`                                  | Calculate probability of detection             |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~horizon_distance_analytical`           | Calculate detector horizon using analytical    |
   |                                                | formula                                        |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~horizon_distance_numerical`            | Calculate detector horizon with optimal sky    |
   |                                                | positioning                                    |
   +------------------------------------------------+------------------------------------------------+

   Instance Attributes
   ----------
   GWSNR class has the following attributes:

   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | Attribute                                      | Type             | Unit  | Description                                    |
   +================================================+==================+=======+================================================+
   | :meth:`~npool`                                 | ``int``          |       | Number of processors for parallel processing   |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~mtot_min`                              | ``float``        | M☉    | Minimum total mass for interpolation grid      |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~mtot_max`                              | ``float``        | M☉    | Maximum total mass for interpolation grid      |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~ratio_min`                             | ``float``        |       | Minimum mass ratio (q = m2/m1)                 |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~ratio_max`                             | ``float``        |       | Maximum mass ratio                             |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~spin_max`                              | ``float``        |       | Maximum aligned spin magnitude                 |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~mtot_resolution`                       | ``int``          |       | Grid resolution for total mass interpolation   |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~ratio_resolution`                      | ``int``          |       | Grid resolution for mass ratio interpolation   |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~spin_resolution`                       | ``int``          |       | Grid resolution for aligned spin interpolation |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~ratio_arr`                             | ``ndarray``      |       | Mass ratio interpolation grid points           |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~mtot_arr`                              | ``ndarray``      | M☉    | Total mass interpolation grid points           |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~a_1_arr`                               | ``ndarray``      |       | Primary aligned spin interpolation grid        |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~a_2_arr`                               | ``ndarray``      |       | Secondary aligned spin interpolation grid      |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~sampling_frequency`                    | ``float``        | Hz    | Detector sampling frequency                    |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~waveform_approximant`                  | ``str``          |       | LAL waveform approximant name                  |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~frequency_domain_source_model`         | ``str``          |       | Bilby frequency domain source model function   |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~f_min`                                 | ``float``        | Hz    | Minimum waveform frequency                     |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~f_ref`                                 | ``float``        | Hz    | Reference frequency for waveform generation    |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~duration_max`                          | ``float/None``   | s     | Maximum waveform duration                      |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~duration_min`                          | ``float/None``   | s     | Minimum waveform duration                      |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_method`                            | ``str``          |       | SNR calculation method                         |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_type`                              | ``str``          |       | SNR type: 'optimal_snr' or 'observed_snr'      |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~noise_realization`                     | ``ndarray/None`` |       | Noise realization for observed SNR             |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~psds_list`                             | ``list``         |       | Detector power spectral densities              |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~detector_tensor_list`                  | ``list``         |       | Detector tensors for antenna response          |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~detector_list`                         | ``list``         |       | Detector names (e.g., ['H1', 'L1', 'V1'])      |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~ifos`                                  | ``list``         |       | Bilby interferometer objects                   |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~interpolator_dir`                      | ``str``          |       | Directory for interpolation coefficients       |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~path_interpolator`                     | ``list``         |       | Paths to interpolation coefficient files       |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_partialsacaled_list`               | ``list``         |       | Partial-scaled SNR interpolation coefficients  |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~multiprocessing_verbose`               | ``bool``         |       | Show progress bars for computations            |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~identifier_dict`                       | ``dict``         |       | Interpolator parameter dictionary for caching  |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_th`                                | ``float``        |       | Individual detector SNR threshold              |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_th_net`                            | ``float``        |       | Network SNR threshold                          |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~model_dict`                            | ``dict``         |       | ANN models for each detector                   |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~scaler_dict`                           | ``dict``         |       | ANN feature scalers for each detector          |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~error_adjustment`                      | ``dict``         |       | ANN error correction parameters                |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~ann_catalogue`                         | ``dict``         |       | ANN model configuration and paths              |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_recalculation`                     | ``bool``         |       | Enable hybrid SNR recalculation                |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_recalculation_range`               | ``list``         |       | SNR range [min, max] triggering recalculation  |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~snr_recalculation_waveform_approximant`| ``str``          |       | Waveform approximant for recalculation         |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~get_interpolated_snr`                  | ``function``     |       | Interpolated SNR calculation function          |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :meth:`~noise_weighted_inner_product_jax`      | ``function``     |       | JAX-accelerated inner product function         |
   +------------------------------------------------+------------------+-------+------------------------------------------------+



   ..
       !! processed by numpydoc !!
   .. py:property:: npool

      
      Number of processors for parallel processing.



      :Returns:

          **npool** : `int`
              Number of processors for parallel processing.

              default: 4













      ..
          !! processed by numpydoc !!

   .. py:property:: duration_max

      
      Maximum waveform duration.



      :Returns:

          **duration_max** : `float` or `None`
              Maximum waveform duration (s). Auto-set if None.

              default: None













      ..
          !! processed by numpydoc !!

   .. py:property:: duration_min

      
      Minimum waveform duration.



      :Returns:

          **duration_min** : `float` or `None`
              Minimum waveform duration (s). Auto-set if None.

              default: None













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_method

      
      SNR calculation method.



      :Returns:

          **snr_method** : `str`
              SNR calculation method. Options: interpolation variants,
              inner_product variants, ann.

              default: 'interpolation_aligned_spins'













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_type

      
      SNR type for calculations.



      :Returns:

          **snr_type** : `str`
              SNR type: 'optimal_snr' or 'observed_snr' (not implemented).

              default: 'optimal_snr'













      ..
          !! processed by numpydoc !!

   .. py:property:: noise_realization

      
      Noise realization for observed SNR.



      :Returns:

          **noise_realization** : `numpy.ndarray` or `None`
              Noise realization for observed SNR (not implemented).

              default: None













      ..
          !! processed by numpydoc !!

   .. py:property:: spin_max

      
      Maximum aligned spin magnitude for interpolation.



      :Returns:

          **spin_max** : `float`
              Maximum aligned spin magnitude for interpolation.

              default: 0.99













      ..
          !! processed by numpydoc !!

   .. py:property:: mtot_max

      
      Maximum total mass for interpolation grid.



      :Returns:

          **mtot_max** : `float`
              Maximum total mass (M☉) for interpolation grid.

              default: 235.0













      ..
          !! processed by numpydoc !!

   .. py:property:: mtot_min

      
      Minimum total mass for interpolation grid.



      :Returns:

          **mtot_min** : `float`
              Minimum total mass (M☉) for interpolation grid.

              default: 9.96













      ..
          !! processed by numpydoc !!

   .. py:property:: ratio_min

      
      Minimum mass ratio for interpolation grid.



      :Returns:

          **ratio_min** : `float`
              Minimum mass ratio (q = m2/m1) for interpolation grid.

              default: 0.1













      ..
          !! processed by numpydoc !!

   .. py:property:: ratio_max

      
      Maximum mass ratio for interpolation grid.



      :Returns:

          **ratio_max** : `float`
              Maximum mass ratio for interpolation grid.

              default: 1.0













      ..
          !! processed by numpydoc !!

   .. py:property:: mtot_resolution

      
      Grid resolution for total mass interpolation.



      :Returns:

          **mtot_resolution** : `int`
              Grid resolution for total mass interpolation.

              default: 200













      ..
          !! processed by numpydoc !!

   .. py:property:: ratio_resolution

      
      Grid resolution for mass ratio interpolation.



      :Returns:

          **ratio_resolution** : `int`
              Grid resolution for mass ratio interpolation.

              default: 20













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_recalculation

      
      Enable hybrid SNR recalculation near detection threshold.



      :Returns:

          **snr_recalculation** : `bool`
              Enable hybrid SNR recalculation near detection threshold.

              default: False













      ..
          !! processed by numpydoc !!

   .. py:property:: ratio_arr

      
      Mass ratio interpolation grid points.



      :Returns:

          **ratio_arr** : `numpy.ndarray`
              Mass ratio interpolation grid points.













      ..
          !! processed by numpydoc !!

   .. py:property:: mtot_arr

      
      Total mass interpolation grid points.



      :Returns:

          **mtot_arr** : `numpy.ndarray`
              Total mass (M☉) interpolation grid points.













      ..
          !! processed by numpydoc !!

   .. py:property:: sampling_frequency

      
      Detector sampling frequency.



      :Returns:

          **sampling_frequency** : `float`
              Detector sampling frequency (Hz).

              default: 2048.0













      ..
          !! processed by numpydoc !!

   .. py:property:: waveform_approximant

      
      LAL waveform approximant name.



      :Returns:

          **waveform_approximant** : `str`
              LAL waveform approximant (e.g., 'IMRPhenomD', 'IMRPhenomXPHM').

              default: 'IMRPhenomD'













      ..
          !! processed by numpydoc !!

   .. py:property:: frequency_domain_source_model

      
      Bilby frequency domain source model function.



      :Returns:

          **frequency_domain_source_model** : `str`
              Bilby frequency domain source model function.

              default: 'lal_binary_black_hole'













      ..
          !! processed by numpydoc !!

   .. py:property:: f_min

      
      Minimum waveform frequency.



      :Returns:

          **f_min** : `float`
              Minimum waveform frequency (Hz).

              default: 20.0













      ..
          !! processed by numpydoc !!

   .. py:property:: f_ref

      
      Reference frequency for waveform generation.



      :Returns:

          **f_ref** : `float`
              Reference frequency (Hz) for waveform generation.

              default: same as f_min













      ..
          !! processed by numpydoc !!

   .. py:property:: interpolator_dir

      
      Directory for interpolation coefficient storage.



      :Returns:

          **interpolator_dir** : `str`
              Directory for interpolation coefficient storage.
              default: './interpolator_json'













      ..
          !! processed by numpydoc !!

   .. py:property:: multiprocessing_verbose

      
      Show progress bars for multiprocessing computations.



      :Returns:

          **multiprocessing_verbose** : `bool`
              Show progress bars for multiprocessing computations.

              default: True













      ..
          !! processed by numpydoc !!

   .. py:property:: spin_resolution

      
      Grid resolution for aligned spin interpolation.



      :Returns:

          **spin_resolution** : `int`
              Grid resolution for aligned spin interpolation.

              default: 10













      ..
          !! processed by numpydoc !!

   .. py:property:: a_1_arr

      
      Primary aligned spin interpolation grid.



      :Returns:

          **a_1_arr** : `numpy.ndarray`
              Primary aligned spin interpolation grid.













      ..
          !! processed by numpydoc !!

   .. py:property:: a_2_arr

      
      Secondary aligned spin interpolation grid.



      :Returns:

          **a_2_arr** : `numpy.ndarray`
              Secondary aligned spin interpolation grid.













      ..
          !! processed by numpydoc !!

   .. py:property:: identifier_dict

      
      Interpolator parameter dictionary for caching.



      :Returns:

          **identifier_dict** : `dict`
              Interpolator parameter dictionary for caching.













      ..
          !! processed by numpydoc !!

   .. py:property:: psds_list

      
      Detector power spectral densities.

      for the i-th detector:

          psds_list[i][0]: frequency (numpy.ndarray)

          psds_list[i][1]: power spectral density (numpy.ndarray)

          psds_list[i][2]: scipy.interpolate.interp1d object


      :Returns:

          **psds_list** : `list`
              List of PowerSpectralDensity objects for each detector.













      ..
          !! processed by numpydoc !!

   .. py:property:: detector_tensor_list

      
      Detector tensors for antenna response calculations.



      :Returns:

          **detector_tensor_list** : `list`
              List of numpy.ndarray detector tensors for antenna response.













      ..
          !! processed by numpydoc !!

   .. py:property:: detector_list

      
      Detector names.



      :Returns:

          **detector_list** : `list`
              List of detector names (e.g., ['H1', 'L1', 'V1']).













      ..
          !! processed by numpydoc !!

   .. py:property:: ifos

      
      Bilby interferometer objects.



      :Returns:

          **ifos** : `list`
              List of Bilby Interferometer objects.













      ..
          !! processed by numpydoc !!

   .. py:property:: path_interpolator

      
      Paths to interpolation coefficient files.



      :Returns:

          **path_interpolator** : `list`
              List of paths to interpolation coefficient files.













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_partialsacaled_list

      
      Partial-scaled SNR interpolation coefficients.



      :Returns:

          **snr_partialsacaled_list** : `list`
              List of numpy.ndarray partial-scaled SNR interpolation coefficients.













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_th

      
      Individual detector SNR threshold.



      :Returns:

          **snr_th** : `float`
              Individual detector SNR threshold.

              default: 10.0













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_th_net

      
      Network SNR threshold.



      :Returns:

          **snr_th_net** : `float`
              Network SNR threshold.

              default: 10.0













      ..
          !! processed by numpydoc !!

   .. py:property:: model_dict

      
      ANN models for each detector.



      :Returns:

          **model_dict** : `dict`
              ANN models for each detector (when snr_method='ann').













      ..
          !! processed by numpydoc !!

   .. py:property:: scaler_dict

      
      ANN feature scalers for each detector.



      :Returns:

          **scaler_dict** : `dict`
              ANN feature scalers for each detector (when snr_method='ann').













      ..
          !! processed by numpydoc !!

   .. py:property:: error_adjustment

      
      ANN error correction parameters.



      :Returns:

          **error_adjustment** : `dict`
              ANN error correction parameters (when snr_method='ann').













      ..
          !! processed by numpydoc !!

   .. py:property:: ann_catalogue

      
      ANN model configuration and paths.



      :Returns:

          **ann_catalogue** : `dict`
              ANN model configuration and paths (when snr_method='ann').













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_recalculation_range

      
      SNR range triggering recalculation.



      :Returns:

          **snr_recalculation_range** : `list`
              SNR range [min, max] triggering recalculation.

              default: [6, 14]













      ..
          !! processed by numpydoc !!

   .. py:property:: snr_recalculation_waveform_approximant

      
      Waveform approximant for SNR recalculation.



      :Returns:

          **snr_recalculation_waveform_approximant** : `str`
              Waveform approximant for SNR recalculation.

              default: 'IMRPhenomXPHM'













      ..
          !! processed by numpydoc !!

   .. py:property:: get_interpolated_snr

      
      Interpolated SNR calculation function.



      :Returns:

          **get_interpolated_snr** : `function`
              Interpolated SNR calculation function (backend-specific).













      ..
          !! processed by numpydoc !!

   .. py:property:: noise_weighted_inner_product_jax

      
      JAX-accelerated inner product function.



      :Returns:

          **noise_weighted_inner_product_jax** : `function`
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

   .. py:method:: optimal_snr(mass_1=np.array([10.0]), mass_2=np.array([10.0]), luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=None, output_jsonfile=False)

      
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

          **gw_param_dict** : dict or None, default=None
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
              (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
              Values are arrays matching input size.








      .. rubric:: Notes

      - For interpolation methods, tilt angles are converted to aligned spins: a_i * cos(tilt_i)
      - Total mass must be within [mtot_min, mtot_max] for non-zero SNR
      - Hybrid recalculation uses higher-order waveforms near detection threshold if enabled
      - Compatible with all configured detector networks and waveform approximants


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='interpolation_no_spins')
      >>> result = snr.optimal_snr(mass_1=30.0, mass_2=25.0, luminosity_distance=100.0)
      >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")

      >>> # Multiple systems with parameter dictionary
      >>> params = {'mass_1': [20, 30], 'mass_2': [20, 25], 'luminosity_distance': [100, 200]}
      >>> result = snr.optimal_snr(gw_param_dict=params)



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_ann(mass_1=30.0, mass_2=29.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, gw_param_dict=None, output_jsonfile=False)

      
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

          **gw_param_dict** : dict or None, default=None
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR estimates for each detector and network. Keys are 'optimal_snr_{detector}'
              (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.








      .. rubric:: Notes

      - Requires pre-trained ANN models loaded during initialization
      - Uses aligned spin components: a_i * cos(tilt_i) for effective spin calculation
      - ANN inputs: partial-scaled SNR, amplitude factor, mass ratio, effective spin, inclination
      - Applies error correction to improve prediction accuracy
      - Total mass must be within [mtot_min, mtot_max] for valid results


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='ann')
      >>> result = snr.optimal_snr_with_ann(mass_1=30, mass_2=25, a_1=0.5, tilt_1=0.2)
      >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_interpolation(mass_1=30.0, mass_2=29.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, output_jsonfile=False, gw_param_dict=None)

      
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

          **gw_param_dict** : dict or None, default=None
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
              (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
              Systems outside mass bounds have zero SNR.








      .. rubric:: Notes

      - Requires precomputed interpolation coefficients from class initialization
      - self.get_interpolated_snr is set based on snr_method (Numba or JAX or MLX) and whether the system is non-spinning or aligned-spin
      - Total mass must be within [mtot_min, mtot_max] for valid results
      - Uses aligned spin: a_i * cos(tilt_i) for spin-enabled methods
      - Backend acceleration available via JAX or Numba depending on snr_method


      .. rubric:: Examples

      >>> snr_calc = GWSNR(snr_method='interpolation_no_spins')
      >>> result = snr_calc.optimal_snr_with_interpolation(mass_1=30, mass_2=25)
      >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_inner_product(mass_1=10, mass_2=10, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=None, output_jsonfile=False)

      
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

          **gw_param_dict** : dict or None, default=None
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
              (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
              Systems outside mass bounds have zero SNR.








      .. rubric:: Notes

      - Waveform duration auto-estimated from chirp time with 1.1x safety factor
      - Uses multiprocessing for parallel computation across npool processors
      - Requires 'if __name__ == "__main__":' guard when using multiprocessing
      - Most accurate method but slower than interpolation for population studies


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product')
      >>> result = snr.optimal_snr_with_inner_product(mass_1=30, mass_2=25)
      >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: optimal_snr_with_inner_product_ripple(mass_1=10, mass_2=10, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=None, output_jsonfile=False)

      
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

          **gw_param_dict** : dict or None, default=None
              Parameter dictionary. If provided, overrides individual arguments.

          **output_jsonfile** : str or bool, default=False
              Save results to JSON file. If True, saves as 'snr.json'.

      :Returns:

          dict
              SNR values for each detector and network SNR. Keys are 'optimal_snr_{detector}'
              (e.g., 'optimal_snr_H1', 'optimal_snr_L1', 'optimal_snr_V1') and 'optimal_snr_net'.
              Systems outside mass bounds have zero SNR.








      .. rubric:: Notes

      - Requires snr_method='inner_product_jax' during initialization
      - Uses JAX JIT compilation and vectorization for GPU acceleration
      - Duration auto-estimated with safety bounds from duration_min/max
      - Compatible with Ripple-supported approximants (IMRPhenomD, IMRPhenomXPHM)
      - Supports precessing spins through full parameter space


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product_jax')
      >>> result = snr.optimal_snr_with_inner_product_ripple(mass_1=30, mass_2=25)
      >>> print(f"Network SNR: {result['optimal_snr_net'][0]:.2f}")



      ..
          !! processed by numpydoc !!

   .. py:method:: pdet(mass_1=np.array([10.0]), mass_2=np.array([10.0]), luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, gw_param_dict=None, output_jsonfile=False, snr_th=None, snr_th_net=None, pdet_type=None, distribution_type=None, include_optimal_snr=False, include_observed_snr=False)

      
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

          **gw_param_dict** : dict or None, default=None
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
              Detection probabilities for each detector and network. Keys are 'pdet_{detector}'
              (e.g., 'pdet_H1', 'pdet_L1', 'pdet_V1') and 'pdet_net'. Values depend on pdet_type:
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

      :Returns:

          **horizon_distance_dict** : dict
              Horizon distances in Mpc for each detector.
              Keys: 'horizon_distance_{detector}' (e.g., 'horizon_distance_H1', 'horizon_distance_L1').
              Values: horizon distance in Mpc for the corresponding detector.








      .. rubric:: Notes

      - Assumes optimal orientation: θ_jn=0, overhead sky location
      - Formula: d_horizon = (d_eff/SNR_th) x SNR_opt
      - Compatible with all waveform approximants
      - Does not calculate network horizon; use horizon_distance_numerical for network


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product')
      >>> horizon = snr.horizon_distance_analytical(mass_1=1.4, mass_2=1.4)
      >>> print(f"H1 horizon: {horizon['horizon_distance_H1']:.1f} Mpc")



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
              Horizon distances in Mpc for each detector and network.
              Keys: 'horizon_distance_{detector}' (e.g., 'horizon_distance_H1', 'horizon_distance_L1')
              and 'horizon_distance_net' for the network.

          **optimal_sky_location** : dict
              Optimal sky coordinates (ra, dec) in radians for maximum SNR at given geocent_time.
              Keys: 'optimal_sky_location_{detector}' (e.g., 'optimal_sky_location_H1')
              and 'optimal_sky_location_net' for the network.








      .. rubric:: Notes

      - Uses differential evolution to find optimal sky location maximizing antenna response
      - Network horizon maximizes quadrature sum of detector SNRs
      - Individual detector horizons maximize (F_plus² + F_cross²)
      - Root finding determines distance where SNR equals threshold
      - Computation time depends on optimization tolerances and system complexity


      .. rubric:: Examples

      >>> snr = GWSNR(snr_method='inner_product')
      >>> horizon, sky = snr.horizon_distance_numerical(mass_1=1.4, mass_2=1.4)
      >>> print(f"Network horizon: {horizon['horizon_distance_net']:.1f} Mpc at (RA={sky['optimal_sky_location_net'][0]:.2f}, Dec={sky['optimal_sky_location_net'][1]:.2f})")



      ..
          !! processed by numpydoc !!


