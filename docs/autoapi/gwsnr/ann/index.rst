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


   
   Generate and train ANN models for gravitational wave SNR prediction.

   Provides functionality to train artificial neural network models that predict
   optimal SNR for gravitational wave signals from compact binary coalescences.
   Uses interpolated partial-scaled SNR values as input features along with
   intrinsic binary parameters.

   Key Features:
   - TensorFlow/Keras-based ANN model training

   - Feature extraction using GWSNR interpolation framework

   - StandardScaler normalization for input features

   - Linear error adjustment for improved prediction accuracy

   - Model evaluation with confusion matrix and accuracy metrics

   :Parameters:

       **directory** : ``str``
           Output directory for saving trained models, scalers, and configurations.

           default: './gwsnr_data'

       **npool** : ``int``
           Number of processors for parallel GWSNR computation.

           default: 4

       **gwsnr_verbose** : ``bool``
           If True, print GWSNR initialization progress.

           default: True

       **snr_th** : ``float``
           SNR threshold for detection classification.

           default: 8.0

       **snr_method** : ``str``
           SNR calculation method for GWSNR initialization.

           default: 'interpolation_aligned_spins'

       **waveform_approximant** : ``str``
           Waveform approximant for SNR calculation and ANN training.

           default: 'IMRPhenomXPHM'

       **\*\*kwargs** : ``dict``
           Additional keyword arguments passed to :class:`~gwsnr.GWSNR`.

           See GWSNR documentation for available options.









   .. rubric:: Notes

   - ANN input features: [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn]

   - Training requires pre-generated GW parameter samples with computed SNR values

   - Error adjustment improves predictions via linear correction: y_adj = y_pred - (a*y_pred + b)

   - Only single-detector training is supported per instance


   .. rubric:: Examples

   Basic ANN model training:

   >>> from gwsnr import ANNModelGenerator
   >>> amg = ANNModelGenerator()
   >>> amg.ann_model_training(gw_param_dict='gw_param_dict.json')

   Custom configuration:

   >>> amg = ANNModelGenerator(
           directory='./custom_output',
           snr_th=10.0,
           waveform_approximant='IMRPhenomD')
   >>> amg.ann_model_training(
           gw_param_dict=params,
           epochs=200,
           batch_size=64)

   Instance Methods
   ----------
   ANNModelGenerator class has the following methods:

   +------------------------------------------------+------------------------------------------------+
   | Method                                         | Description                                    |
   +================================================+================================================+
   | :meth:`~ann_model_training`                    | Train ANN model with parameter data            |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~load_model_scaler_error`               | Load pre-trained model, scaler, and error data |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~predict_snr`                           | Predict SNR using trained ANN model            |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~predict_pdet`                          | Predict detection probability using trained    |
   |                                                | ANN model                                      |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~pdet_error`                            | Calculate detection probability error rate     |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~pdet_confusion_matrix`                 | Generate confusion matrix for Pdet evaluation  |
   +------------------------------------------------+------------------------------------------------+
   | :meth:`~snr_error_adjustment`                  | Update and save error adjustment parameters    |
   +------------------------------------------------+------------------------------------------------+

   Instance Attributes
   ----------
   ANNModelGenerator class has the following attributes:

   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | Attribute                                      | Type             | Unit  | Description                                    |
   +================================================+==================+=======+================================================+
   | :attr:`~directory`                             | ``str``          |       | Output directory for model files               |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~ann_model`                             | ``function``     |       | ANN model constructor function                 |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~ann`                                   | ``Model/None``   |       | Trained Keras model instance                   |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~scaler`                                | ``Scaler/None``  |       | StandardScaler for feature normalization       |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~gwsnr_args`                            | ``dict``         |       | GWSNR initialization arguments                 |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~gwsnr`                                 | ``GWSNR``        |       | GWSNR instance for interpolation               |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~X_test`                                | ``ndarray``      |       | Scaled test input features                     |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~y_test`                                | ``ndarray``      |       | Test output labels (SNR values)                |
   +------------------------------------------------+------------------+-------+------------------------------------------------+
   | :attr:`~error_adjustment`                      | ``dict``         |       | Error correction parameters (slope, intercept) |
   +------------------------------------------------+------------------+-------+------------------------------------------------+



   ..
       !! processed by numpydoc !!
   .. py:property:: directory

      
      Output directory for model files.



      :Returns:

          **directory** : ``str``
              Output directory path for saving trained models, scalers, and configurations.

              default: './gwsnr_data'













      ..
          !! processed by numpydoc !!

   .. py:property:: ann_model

      
      ANN model constructor function.



      :Returns:

          **ann_model** : ``function``
              Function that creates and compiles a Keras Sequential model.













      ..
          !! processed by numpydoc !!

   .. py:property:: ann

      
      Trained Keras model instance.



      :Returns:

          **ann** : ``tensorflow.keras.Model`` or ``None``
              Trained ANN model, or None if not yet trained/loaded.













      ..
          !! processed by numpydoc !!

   .. py:property:: scaler

      
      StandardScaler for feature normalization.



      :Returns:

          **scaler** : ``sklearn.preprocessing.StandardScaler`` or ``None``
              Fitted scaler for input feature normalization, or None if not fitted.













      ..
          !! processed by numpydoc !!

   .. py:property:: gwsnr_args

      
      GWSNR initialization arguments.



      :Returns:

          **gwsnr_args** : ``dict``
              Dictionary of GWSNR configuration parameters.













      ..
          !! processed by numpydoc !!

   .. py:property:: gwsnr

      
      GWSNR instance for interpolation.



      :Returns:

          **gwsnr** : ``GWSNR``
              GWSNR instance used for partial-scaled SNR interpolation.













      ..
          !! processed by numpydoc !!

   .. py:property:: X_test

      
      Scaled test input features.



      :Returns:

          **X_test** : ``numpy.ndarray`` or ``None``
              Scaled test input feature array, or None if not set.













      ..
          !! processed by numpydoc !!

   .. py:property:: y_test

      
      Test output labels (SNR values).



      :Returns:

          **y_test** : ``numpy.ndarray`` or ``None``
              Test output SNR values array, or None if not set.













      ..
          !! processed by numpydoc !!

   .. py:property:: error_adjustment

      
      Error correction parameters.



      :Returns:

          **error_adjustment** : ``dict`` or ``None``
              Dictionary with 'slope' and 'intercept' keys for linear error correction,
              or None if not computed.













      ..
          !! processed by numpydoc !!

   .. py:method:: ann_model_training(gw_param_dict, randomize=True, test_size=0.1, random_state=42, num_nodes_list=[5, 32, 32, 1], activation_fn_list=['relu', 'relu', 'sigmoid', 'linear'], optimizer='adam', loss='mean_squared_error', metrics=['accuracy'], batch_size=32, epochs=100, error_adjustment_snr_range=[4, 10], ann_file_name='ann_model.h5', scaler_file_name='scaler.pkl', error_adjustment_file_name='error_adjustment.json', ann_path_dict_file_name='ann_path_dict.json')

      
      Train ANN model for SNR prediction using GW parameter data.

      Complete training pipeline including data preparation, model training,
      error adjustment calculation, and file saving.

      :Parameters:

          **gw_param_dict** : ``dict`` or ``str``
              GW parameter dictionary or path to JSON file containing training data.

          **randomize** : ``bool``
              If True, randomly shuffle the training data.

              default: True

          **test_size** : ``float``
              Fraction of data to hold out for testing.

              default: 0.1

          **random_state** : ``int``
              Random state for train/test split reproducibility.

              default: 42

          **num_nodes_list** : ``list`` of ``int``
              Number of nodes in each layer.

              default: [5, 32, 32, 1]

          **activation_fn_list** : ``list`` of ``str``
              Activation functions for each layer.

              default: ['relu', 'relu', 'sigmoid', 'linear']

          **optimizer** : ``str``
              Keras optimizer name.

              default: 'adam'

          **loss** : ``str``
              Keras loss function name.

              default: 'mean_squared_error'

          **metrics** : ``list`` of ``str``
              Metrics to evaluate during training.

              default: ['accuracy']

          **batch_size** : ``int``
              Batch size for training.

              default: 32

          **epochs** : ``int``
              Number of training epochs.

              default: 100

          **error_adjustment_snr_range** : ``list`` of ``float``
              SNR range [min, max] for computing error adjustment parameters.

              default: [4, 10]

          **ann_file_name** : ``str``
              Output filename for trained model.

              default: 'ann_model.h5'

          **scaler_file_name** : ``str``
              Output filename for fitted scaler.

              default: 'scaler.pkl'

          **error_adjustment_file_name** : ``str``
              Output filename for error adjustment parameters.

              default: 'error_adjustment.json'

          **ann_path_dict_file_name** : ``str``
              Output filename for ANN configuration paths.

              default: 'ann_path_dict.json'









      .. rubric:: Notes

      - Saves model to :attr:`~directory`/{ann_file_name}

      - Saves scaler to :attr:`~directory`/{scaler_file_name}

      - Computes error adjustment using :meth:`~_helper_error_adjustment`

      - Stores test data in :attr:`~X_test` and :attr:`~y_test`


      .. rubric:: Examples

      >>> amg = ANNModelGenerator()
      >>> amg.ann_model_training(
              gw_param_dict='training_params.json',
              epochs=200,
              batch_size=64)



      ..
          !! processed by numpydoc !!

   .. py:method:: pdet_error(gw_param_dict=None, randomize=True, error_adjustment=True)

      
      Calculate detection probability error rate.

      Evaluates the percentage of samples where predicted and true detection
      status (SNR > threshold) differ.

      :Parameters:

          **gw_param_dict** : ``dict`` or ``str``
              GW parameter dictionary or JSON file path. If None, uses stored test data. Optional.

          **randomize** : ``bool``
              If True, randomly shuffle parameters (only used if gw_param_dict provided).

              default: True

          **error_adjustment** : ``bool``
              If True, apply linear error correction to predictions.

              default: True

      :Returns:

          **error** : ``float``
              Percentage of misclassified samples.

          **y_pred** : ``numpy.ndarray``
              Predicted SNR values (with or without error adjustment).

          **y_test** : ``numpy.ndarray``
              True SNR values.








      .. rubric:: Notes

      - Uses :attr:`~gwsnr_args['snr_th']` as detection threshold

      - Error adjustment: y_adj = y_pred - (slope*y_pred + intercept)





      ..
          !! processed by numpydoc !!

   .. py:method:: pdet_confusion_matrix(gw_param_dict=None, randomize=True, snr_threshold=8.0)

      
      Generate confusion matrix for detection probability classification.

      Evaluates ANN predictions as binary classification (detected/not detected)
      and computes confusion matrix and accuracy metrics.

      :Parameters:

          **gw_param_dict** : ``dict`` or ``str``
              GW parameter dictionary or JSON file path. If None, uses stored test data. Optional.

          **randomize** : ``bool``
              If True, randomly shuffle parameters (only used if gw_param_dict provided).

              default: True

          **snr_threshold** : ``float``
              SNR threshold for detection classification.

              default: 8.0

      :Returns:

          **cm** : ``numpy.ndarray``
              Confusion matrix of shape (2, 2).

          **accuracy** : ``float``
              Classification accuracy percentage.

          **y_pred** : ``numpy.ndarray``
              Predicted detection status (boolean array).

          **y_test** : ``numpy.ndarray``
              True detection status (boolean array).








      .. rubric:: Notes

      - Uses sklearn.metrics.confusion_matrix and accuracy_score

      - Prints confusion matrix and accuracy to stdout





      ..
          !! processed by numpydoc !!

   .. py:method:: load_model_scaler_error(ann_file_name='ann_model.h5', scaler_file_name='scaler.pkl', error_adjustment_file_name=False)

      
      Load pre-trained ANN model, scaler, and optionally error adjustment.

      Restores saved model components for prediction use.

      :Parameters:

          **ann_file_name** : ``str``
              Filename of the trained model.

              default: 'ann_model.h5'

          **scaler_file_name** : ``str``
              Filename of the fitted scaler.

              default: 'scaler.pkl'

          **error_adjustment_file_name** : ``str`` or ``bool``
              Filename of error adjustment parameters. If False, not loaded.

              default: False

      :Returns:

          **ann** : ``tensorflow.keras.Model``
              Loaded Keras model.

          **scaler** : ``sklearn.preprocessing.StandardScaler``
              Loaded scaler.

          **error_adjustment** : ``dict``
              Error adjustment parameters (only returned if error_adjustment_file_name is provided). Optional.








      .. rubric:: Notes

      - Updates :attr:`~ann`, :attr:`~scaler`, and optionally :attr:`~error_adjustment`

      - Files are loaded from :attr:`~directory`





      ..
          !! processed by numpydoc !!

   .. py:method:: snr_error_adjustment(gw_param_dict=None, randomize=True, snr_range=[4, 10], error_adjustment_file_name='error_adjustment.json')

      
      Recalculate and save error adjustment parameters.

      Computes new error adjustment based on current predictions and updates
      the stored parameters.

      :Parameters:

          **gw_param_dict** : ``dict`` or ``str``
              GW parameter dictionary or JSON file path for evaluation. Optional.

          **randomize** : ``bool``
              If True, randomly shuffle parameters.

              default: True

          **snr_range** : ``list`` of ``float``
              SNR range for error adjustment fitting.

              default: [4, 10]

          **error_adjustment_file_name** : ``str``
              Output filename for updated error adjustment.

              default: 'error_adjustment.json'

      :Returns:

          **error_adjustment** : ``dict``
              Updated error adjustment parameters with 'slope' and 'intercept'.








      .. rubric:: Notes

      - Calls :meth:`~pdet_error` with error_adjustment=True for predictions

      - Saves updated parameters to :attr:`~directory`/{error_adjustment_file_name}





      ..
          !! processed by numpydoc !!

   .. py:method:: predict_snr(gw_param_dict, error_adjustment=True)

      
      Predict SNR values using trained ANN model.

      Applies the trained model to new GW parameters and optionally
      applies error correction.

      :Parameters:

          **gw_param_dict** : ``dict`` or ``str``
              GW parameter dictionary or path to JSON file.

          **error_adjustment** : ``bool``
              If True, apply linear error correction to predictions.

              default: True

      :Returns:

          **y_pred** : ``numpy.ndarray``
              Predicted SNR values (corrected if error_adjustment=True).








      .. rubric:: Notes

      - Requires :attr:`~ann` and :attr:`~scaler` to be loaded/trained

      - Error adjustment: y_adj = y_pred - (slope*y_pred + intercept)





      ..
          !! processed by numpydoc !!

   .. py:method:: predict_pdet(gw_param_dict, snr_threshold=8.0, error_adjustment=True)

      
      Predict detection probability using trained ANN model.

      Classifies events as detected (SNR > threshold) or not detected
      based on ANN predictions.

      :Parameters:

          **gw_param_dict** : ``dict`` or ``str``
              GW parameter dictionary or path to JSON file.

          **snr_threshold** : ``float``
              SNR threshold for detection classification.

              default: 8.0

          **error_adjustment** : ``bool``
              If True, apply error correction before thresholding.

              default: True

      :Returns:

          **y_pred** : ``numpy.ndarray`` of ``bool``
              Detection status for each sample (True = detected).








      .. rubric:: Notes

      - Calls :meth:`~predict_snr` for SNR prediction

      - Returns boolean array: y_pred > snr_threshold





      ..
          !! processed by numpydoc !!


