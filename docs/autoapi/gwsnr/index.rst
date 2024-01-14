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




.. py:class:: GWSNR(npool=int(4), mtot_min=2.0, mtot_max=439.6, ratio_min=0.1, ratio_max=1.0, mtot_resolution=500, ratio_resolution=50, sampling_frequency=2048.0, waveform_approximant='IMRPhenomD', minimum_frequency=20.0, snr_type='interpolation', psds=None, isit_psd_file=False, ifos=None, interpolator_dir='./interpolator_pickle', create_new_interpolator=False, gwsnr_verbose=True, multiprocessing_verbose=True, mtot_cut=True)


   
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
           options: 'interpolation', 'inner_product', 'pdet'

       **psds** : `dict`
           Dictionary of psds for different detectors. Default is None. If None, bilby's default psds will be used. Other options:

           Example 1: when values are psd name from pycbc analytical psds, psds={'L1':'aLIGOaLIGODesignSensitivityT1800044','H1':'aLIGOaLIGODesignSensitivityT1800044','V1':'AdvVirgo'}. To check available psd name run

           >>> import pycbc.psd
           >>> pycbc.psd.get_lalsim_psd_list()
           Example 2: when values are psd txt file available in bilby,
           psds={'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt', 'K1':'KAGRA_design_asd.txt'}.
           For other psd files, check https://github.com/lscsoft/bilby/tree/master/bilby/gw/detector/noise_curves

           Example 3: when values are custom psd txt file. psds={'L1':'custom_psd.txt','H1':'custom_psd.txt'}. Custom created txt file has two columns. 1st column: frequency array, 2nd column: strain.

       **isit_psd_file** : `bool` or `dict`
           If set True, the given value of psds param should be of psds instead of asd. If asd, set isit_psd_file=False. Default is False. If dict, it should be of the form {'L1':True, 'H1':True, 'V1':True} and should have keys for all the detectors.

       **psd_with_time** : `bool` or `float`
           gps end time of strain data for which psd will be found. (this param will be given highest priority), example: psd_with_time=1246527224.169434. If False, psds given in psds param will be used. Default is False. If True (without gps time), psds will be calculated from strain data by setting gps end time as geocent_time-duration. Default is False.

       **ifos** : `list` or `None`
           List of interferometer objects. Default is None. If None, bilby's default interferometer objects will be used. For example for LIGO India detector, it can be defined as follows,

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
           >>> snr = GWSNR(psds=dict(LIO='your_asd_file.txt'), ifos=[ifosLIO], isit_psd_file=[False])

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
   |:meth:`~init_halfscaled`             | Generates halfscaled SNR         |
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

   .. py:attribute:: snr_halfsacaled

      
      ``numpy.ndarray``

      Array of half scaled SNR interpolation coefficients.















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

   .. py:attribute:: isit_psd_file

      
      ``dict``

      dict keys with detector names and values as bool.















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

   .. py:method:: snr(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, gw_param_dict=False, output_jsonfile=False)

      
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
      >>> snr = GWSNR(snrs_type='interpolation')
      >>> snr.snr_with_interpolation(mass_1=10.0, mass_2=10.0, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)



      ..
          !! processed by numpydoc !!

   .. py:method:: init_halfscaled()

      
      Function to generate halfscaled SNR interpolation coefficients. It will save the interpolator in the pickle file path indicated by the path_interpolator attribute.
















      ..
          !! processed by numpydoc !!

   .. py:method:: compute_bilby_snr(mass_1, mass_2, luminosity_distance=100.0, theta_jn=0.0, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, output_jsonfile=False)

      
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

   .. py:method:: pdet(snr_dict, rho_th=8.0, rho_net_th=8.0)

      
      Probaility of detection of GW for the given sensitivity of the detectors


      :Parameters:

          **snr_dict** : `dict`
              Dictionary of SNR for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=snr_arrays).

          **rho_th** : `float`
              Threshold SNR for detection. Default is 8.0.

          **rho_net_th** : `float`
              Threshold net SNR for detection. Default is 8.0.

      :Returns:

          **pdet_dict** : `dict`
              Dictionary of probability of detection for each detector and net SNR (dict.keys()=detector_names and optimal_snr_net, dict.values()=pdet_arrays).













      ..
          !! processed by numpydoc !!

   .. py:method:: detector_horizon(mass_1=1.4, mass_2=1.4, snr_threshold=8.0)

      
      Function for finding detector horizon distance for BNS (m1=m2=1.4)


      :Parameters:

          **mass_1** : `float`
              Primary mass of the binary in solar mass. Default is 1.4.

          **mass_2** : `float`
              Secondary mass of the binary in solar mass. Default is 1.4.

          **snr_threshold** : `float`
              SNR threshold for detection. Default is 8.0.

      :Returns:

          **horizon** : `dict`
              Dictionary of horizon distance for each detector (dict.keys()=detector_names, dict.values()=horizon_distance).













      ..
          !! processed by numpydoc !!


