:py:mod:`gwsnr.utils.multiprocessing_routine`
=============================================

.. py:module:: gwsnr.utils.multiprocessing_routine

.. autoapi-nested-parse::

   Helper functions for multiprocessing in snr generation

   ..
       !! processed by numpydoc !!


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.utils.multiprocessing_routine.noise_weighted_inner_prod_h_inner_h_slim
   gwsnr.utils.multiprocessing_routine.noise_weighted_inner_prod_h_inner_h
   gwsnr.utils.multiprocessing_routine.noise_weighted_inner_prod_d_inner_h
   gwsnr.utils.multiprocessing_routine.noise_weighted_inner_prod_ripple



.. py:function:: noise_weighted_inner_prod_h_inner_h_slim(params)

   
   Optimized version of noise_weighted_inner_prod_h_inner_h that uses shared worker data.

   This function accesses shared data (psd_list, approximant, etc.) from global
   _worker_shared_data instead of receiving it in params. This dramatically reduces
   the amount of data pickled per work item.

   :Parameters:

       **params** : tuple
           Tuple containing only per-work-item data:
           (mass_1, mass_2, luminosity_distance, theta_jn, psi, phase, ra, dec,
            geocent_time, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1,
            lambda_2, eccentricity, duration, iteration_index)

   :Returns:

       tuple
           (hp_inner_hp_list, hc_inner_hc_list, iteration_index)













   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_prod_h_inner_h(params)

   
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
               lambda_1
           params[16] : float
               lambda_2
           params[17] : float
               eccentricity
           params[18] : float
               approximant
           params[19] : float
               f_min
           params[20] : float
               f_ref
           params[21] : float
               duration
           params[22] : float
               sampling_frequency
           params[23] : int
               index tracker
           params[24] : list
               list of psds for each detector
           params[25] : str
               frequency_domain_source_model name

   :Returns:

       **SNRs_list** : list
           contains opt_snr for each detector and net_opt_snr

       **params[22]** : int
           index tracker













   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_prod_d_inner_h(params)

   
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
               lambda_1
           params[16] : float
               lambda_2
           params[17] : float
               eccentricity
           params[18] : float
               approximant
           params[19] : float
               f_min
           params[20] : float
               f_ref
           params[21] : float
               duration
           params[22] : float
               sampling_frequency
           params[23] : int
               index tracker
           params[24] : list
               list of psds for each detector
           params[25] : str
               frequency_domain_source_model name
           params[26] : list or None
               noise realization. If None, then PSD as noise realization

   :Returns:

       **SNRs_list** : list
           contains opt_snr for each detector and net_opt_snr

       **params[22]** : int
           index tracker













   ..
       !! processed by numpydoc !!

.. py:function:: noise_weighted_inner_prod_ripple(params)

   
   Probaility of detection of GW for the given sensitivity of the detectors


   :Parameters:

       **params** : list
           list of parameters for the inner product calculation
           List contains:

           params[0] : `numpy.ndarray`
               plus polarization
           params[1] : `numpy.ndarray`
               cross polarization
           params[2] : `numpy.ndarray`
               frequency array
           params[3] : `float`
               cutt-off size of given arrays
           params[4] : `float`
               minimum frequency
           params[5] : `float`
               duration
           params[6] : `int`
               index
           params[7] : `list`
               psd objects of given detectors

   :Returns:

       **SNRs_list** : list
           contains opt_snr for each detector and net_opt_snr

       **params[22]** : int
           index tracker













   ..
       !! processed by numpydoc !!

