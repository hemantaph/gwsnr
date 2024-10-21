:py:mod:`gwsnr.multiprocessing_routine`
=======================================

.. py:module:: gwsnr.multiprocessing_routine

.. autoapi-nested-parse::

   Helper functions for multiprocessing in snr generation

   ..
       !! processed by numpydoc !!


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.multiprocessing_routine.noise_weighted_inner_prod



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

