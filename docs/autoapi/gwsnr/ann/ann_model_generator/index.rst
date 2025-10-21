:py:mod:`gwsnr.ann.ann_model_generator`
=======================================

.. py:module:: gwsnr.ann.ann_model_generator

.. autoapi-nested-parse::

   This module contains the ANNModelGenerator class which is used to generate the ANN (Artificial Neural Network) model that can be used to predict the SNR of the GW events.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr.ann.ann_model_generator.ANNModelGenerator




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
   .. py:method:: get_input_output_data(params=None, randomize=True)


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

   .. py:method:: standard_scaling_initialization(X_train)


   .. py:method:: ann_model_4layers(num_nodes_list, activation_fn_list, optimizer, loss, metrics)


   .. py:method:: get_parameters(gw_param_dict)


   .. py:method:: get_scaled_data(gw_param_dict, randomize=True, test_size=0.1, random_state=42)


   .. py:method:: ann_model_training(gw_param_dict, randomize=True, test_size=0.1, random_state=42, num_nodes_list=[5, 32, 32, 1], activation_fn_list=['relu', 'relu', 'sigmoid', 'linear'], optimizer='adam', loss='mean_squared_error', metrics=['accuracy'], batch_size=32, epochs=100, error_adjustment_snr_range=[4, 10], ann_file_name='ann_model.h5', scaler_file_name='scaler.pkl', error_adjustment_file_name='error_adjustment.json', ann_path_dict_file_name='ann_path_dict.json')


   .. py:method:: pdet_error(gw_param_dict=None, randomize=True, error_adjustment=True)


   .. py:method:: save_ann_path_dict(ann_file_name='ann_model.h5', scaler_file_name='scaler.pkl', error_adjustment_file_name='error_adjustment.json', ann_path_dict_file_name='ann_path_dict.json')


   .. py:method:: pdet_confusion_matrix(gw_param_dict=None, randomize=True, snr_threshold=8.0)


   .. py:method:: load_model_scaler_error(ann_file_name='ann_model.h5', scaler_file_name='scaler.pkl', error_adjustment_file_name=False)


   .. py:method:: helper_error_adjustment(y_pred, y_test, snr_range=[4, 10])


   .. py:method:: snr_error_adjustment(gw_param_dict=None, randomize=True, snr_threshold=8.0, snr_range=[4, 10], error_adjustment_file_name='error_adjustment.json')


   .. py:method:: predict_snr(gw_param_dict, error_adjustment=True)


   .. py:method:: predict_pdet(gw_param_dict, snr_threshold=8.0, error_adjustment=True)



