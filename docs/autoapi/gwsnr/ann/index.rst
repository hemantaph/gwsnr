:orphan:

:py:mod:`gwsnr.ann`
===================

.. py:module:: gwsnr.ann


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   ann_model_generator/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.ann.ANNModelGenerator




.. py:class:: ANNModelGenerator(directory='./gwsnr_data', npool=4, gwsnr_verbose=True, snr_th=8.0, snr_method='interpolation_aligned_spins', waveform_approximant='IMRPhenomXPHM', **kwargs)


   
   ANNModelGenerator class is used to generate the ANN model that can be used to predict the SNR of the GW events.


   :Parameters:

       **npool** : `int`
           Number of processors to use for parallel processing.
           Default is 4.

       **gwsnr_verbose** : `bool`
           If True, print the progress of the GWSNR calculation.
           Default is True.

       **snr_th** : `float`
           SNR threshold for the error calculation.
           Default is 8.0.

       **waveform_approximant** : `str`
           Waveform approximant to be used for the GWSNR calculation and the ANN model.
           Default is "IMRPhenomXPHM".

       **\*\*kwargs** : `dict`
           Keyword arguments for the GWSNR class.
           To see the list of available arguments,
           >>> from gwsnr import GWSNR
           >>> help(GWSNR)











   .. rubric:: Examples

   >>> from gwsnr import ANNModelGenerator
   >>> amg = ANNModelGenerator()
   >>> amg.ann_model_training(gw_param_dict='gw_param_dict.json') # training the ANN model with pre-generated parameter points



   ..
       !! processed by numpydoc !!
   .. py:method:: get_input_data(params)

      
      Function to generate input and output data for the neural network

      Parameters:
      idx: index of the parameter points
      params: dictionary of parameter points
          params.keys() = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'psi', 'geocent_time', 'ra', 'dec', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'L1']

      Returns:
      X: input data, [snr_partial_[0], amp0[0], eta, chi_eff, theta_jn]
      y: output data, [L1]















      ..
          !! processed by numpydoc !!


