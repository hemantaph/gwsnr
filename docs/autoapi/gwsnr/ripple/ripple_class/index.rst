:orphan:

:py:mod:`gwsnr.ripple.ripple_class`
===================================

.. py:module:: gwsnr.ripple.ripple_class


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.ripple.ripple_class.RippleInnerProduct




.. py:class:: RippleInnerProduct(waveform_name='IMRPhenomD', minimum_frequency=20.0, sampling_frequency=2048.0, reference_frequency=None)


   
   Class to compute the noise weighted inner product for a given waveform and PSD
















   ..
       !! processed by numpydoc !!
   .. py:method:: arg_selection(waveform_name)

      
      Returns the list of arguments required for the chosen waveform.


      :Parameters:

          **waveform_name: `str`**
              The name of the waveform to use. Ripple supported waveforms only.

      :Returns:

          list: List of arguments required for the chosen waveform.
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: select_waveform(waveform_name)

      
      Imports and returns the specified waveform from ripple.waveforms.

      Parameters:
      waveform_name (str): The name of the waveform to import.

      Returns:
      class: The waveform class from ripple.waveforms.















      ..
          !! processed by numpydoc !!

   .. py:method:: noise_weighted_inner_product_jax(gw_param_dict, psd_list, detector_list, duration=None, duration_min=2, duration_max=128, npool=4, multiprocessing_verbose=True)

      
      Compute the noise weighted inner product for a given waveform and PSD.


      :Parameters:

          **gw_param_dict: `dict`**
              Dictionary containing the waveform parameters. The keys should be the parameter names and the values should be numpy arrays.

          **psd_dict: bilby.gw.detector.PowerSpectralDensity object**
              Dictionary containing the power spectral density for each detector.

          **duration: `float` or `numpy.ndarray`**
              Duration of the waveform.
              Default is None. It will compute the duration based on the chirp time.

          **duration_min: `float`**
              Minimum duration of the waveform.
              Default is 2s.

          **duration_max: `float`**
              Maximum duration of the waveform.
              Default is 512s.

          **verbose: `bool`**
              If True, print the waveform parameters and PSDs.
              Default is False.

      :Returns:

          hp_inner_hp: `numpy.ndarray`
              Noise weighted inner product of h+ with h+













      ..
          !! processed by numpydoc !!


