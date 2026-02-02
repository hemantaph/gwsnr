# -*- coding: utf-8 -*-
"""
ANN Model Generator for Gravitational Wave SNR Prediction.

This module provides the ANNModelGenerator class for training artificial neural network (ANN) 
models to predict gravitational wave signal-to-noise ratios (SNR). It enables fast training 
of custom ANN models for SNR prediction, supporting various detector configurations and 
waveform approximants. It integrates with the GWSNR interpolation framework for feature 
extraction and provides tools for model evaluation and error correction.

Key Features:
- ANN model training for single-detector SNR prediction \n
- Feature extraction using interpolated partial-scaled SNR values \n
- StandardScaler normalization for input features \n
- Linear error adjustment for improved prediction accuracy \n
- Confusion matrix and accuracy evaluation for detection classification \n
- Model, scaler, and configuration persistence \n

Copyright (C) 2025 Hemantakumar Phurailatpam and Otto Hannuksela. 
Distributed under MIT License.
"""

import numpy as np
import os
import pickle
from ..core import GWSNR
from ..utils import append_json, get_param_from_json, load_json, load_ann_h5
from scipy.optimize import curve_fit
# import jax
# jax.config.update("jax_enable_x64", True)

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class ANNModelGenerator():
    """
    Generate and train ANN models for gravitational wave SNR prediction.

    Provides functionality to train artificial neural network models that predict
    optimal SNR for gravitational wave signals from compact binary coalescences.
    Uses interpolated partial-scaled SNR values as input features along with
    intrinsic binary parameters.

    Key Features:
    - TensorFlow/Keras-based ANN model training \n
    - Feature extraction using GWSNR interpolation framework \n
    - StandardScaler normalization for input features \n
    - Linear error adjustment for improved prediction accuracy \n
    - Model evaluation with confusion matrix and accuracy metrics \n

    Parameters
    ----------
    directory : ``str``
        Output directory for saving trained models, scalers, and configurations. \n
        default: './gwsnr_data'
    npool : ``int``
        Number of processors for parallel GWSNR computation. \n
        default: 4
    gwsnr_verbose : ``bool``
        If True, print GWSNR initialization progress. \n
        default: True
    snr_th : ``float``
        SNR threshold for detection classification. \n
        default: 8.0
    snr_method : ``str``
        SNR calculation method for GWSNR initialization. \n
        default: 'interpolation_aligned_spins'
    waveform_approximant : ``str``
        Waveform approximant for SNR calculation and ANN training. \n
        default: 'IMRPhenomXPHM'
    **kwargs : ``dict``
        Additional keyword arguments passed to :class:`~gwsnr.GWSNR`. \n
        See GWSNR documentation for available options.

    Examples
    ----------
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
    ANNModelGenerator class has the following methods:  \n
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
    ANNModelGenerator class has the following attributes:  \n
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

    Notes
    ----------
    - ANN input features: [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn] \n
    - Training requires pre-generated GW parameter samples with computed SNR values \n
    - Error adjustment improves predictions via linear correction: y_adj = y_pred - (a*y_pred + b) \n
    - Only single-detector training is supported per instance \n
    """
    
    def __init__(self,
        directory="./gwsnr_data",
        npool=4,
        gwsnr_verbose=True,
        snr_th=8.0,
        snr_method="interpolation_aligned_spins",
        waveform_approximant="IMRPhenomXPHM",
        **kwargs,
    ):
        # Output directory setup
        self._directory = directory
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)

        # ANN model and scaler initialization
        self._ann_model = self._ann_model_4layers
        self._ann = None
        self._scaler = None
        self._X_test = None
        self._y_test = None
        self._error_adjustment = None

        # GWSNR configuration dictionary
        self._gwsnr_args = dict(
            npool=npool,
            mtot_min=2*4.98,  # 4.98 M☉ is minimum component mass in GWTC-3
            mtot_max=2*112.5+10.0,  # 112.5 M☉ is maximum + 10.0 buffer
            ratio_min=0.1,
            ratio_max=1.0,
            spin_max=0.99,
            mtot_resolution=200,
            ratio_resolution=20,
            spin_resolution=10,
            sampling_frequency=2048.0,
            waveform_approximant=waveform_approximant,
            minimum_frequency=20.0,
            snr_method="interpolation_aligned_spins",
            psds=None,
            ifos=None,
            interpolator_dir="./interpolator_json",
            create_new_interpolator=False,
            gwsnr_verbose=gwsnr_verbose,
            multiprocessing_verbose=True,
            mtot_cut=False,
            snr_th=snr_th,
        )
        self._gwsnr_args.update(kwargs)

        # GWSNR instance initialization
        self._gwsnr = GWSNR(
            npool=self._gwsnr_args['npool'],
            mtot_min=self._gwsnr_args['mtot_min'],
            mtot_max=self._gwsnr_args['mtot_max'],
            ratio_min=self._gwsnr_args['ratio_min'],
            ratio_max=self._gwsnr_args['ratio_max'],
            spin_max=self._gwsnr_args['spin_max'],
            mtot_resolution=self._gwsnr_args['mtot_resolution'],
            ratio_resolution=self._gwsnr_args['ratio_resolution'],
            spin_resolution=self._gwsnr_args['spin_resolution'],
            sampling_frequency=self._gwsnr_args['sampling_frequency'],
            waveform_approximant=self._gwsnr_args['waveform_approximant'],
            minimum_frequency=self._gwsnr_args['minimum_frequency'],
            snr_method=snr_method,
            psds=self._gwsnr_args['psds'],
            ifos=self._gwsnr_args['ifos'],
            interpolator_dir=self._gwsnr_args['interpolator_dir'],
            create_new_interpolator=self._gwsnr_args['create_new_interpolator'],
            gwsnr_verbose=self._gwsnr_args['gwsnr_verbose'],
            multiprocessing_verbose=self._gwsnr_args['multiprocessing_verbose'],
            mtot_cut=self._gwsnr_args['mtot_cut'],
            snr_th=self._gwsnr_args['snr_th'],
        )

    # -------------------------------------------
    # PROPERTIES
    # -------------------------------------------
    @property
    def directory(self):
        """
        Output directory for model files.

        Returns
        -------
        directory : ``str``
            Output directory path for saving trained models, scalers, and configurations. \n
            default: './gwsnr_data'
        """
        return self._directory

    @directory.setter
    def directory(self, value):
        self._directory = value

    @property
    def ann_model(self):
        """
        ANN model constructor function.

        Returns
        -------
        ann_model : ``function``
            Function that creates and compiles a Keras Sequential model.
        """
        return self._ann_model

    @ann_model.setter
    def ann_model(self, value):
        self._ann_model = value

    @property
    def ann(self):
        """
        Trained Keras model instance.

        Returns
        -------
        ann : ``tensorflow.keras.Model`` or ``None``
            Trained ANN model, or None if not yet trained/loaded.
        """
        return self._ann

    @ann.setter
    def ann(self, value):
        self._ann = value

    @property
    def scaler(self):
        """
        StandardScaler for feature normalization.

        Returns
        -------
        scaler : ``sklearn.preprocessing.StandardScaler`` or ``None``
            Fitted scaler for input feature normalization, or None if not fitted.
        """
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def gwsnr_args(self):
        """
        GWSNR initialization arguments.

        Returns
        -------
        gwsnr_args : ``dict``
            Dictionary of GWSNR configuration parameters.
        """
        return self._gwsnr_args

    @gwsnr_args.setter
    def gwsnr_args(self, value):
        self._gwsnr_args = value

    @property
    def gwsnr(self):
        """
        GWSNR instance for interpolation.

        Returns
        -------
        gwsnr : ``GWSNR``
            GWSNR instance used for partial-scaled SNR interpolation.
        """
        return self._gwsnr

    @gwsnr.setter
    def gwsnr(self, value):
        self._gwsnr = value

    @property
    def X_test(self):
        """
        Scaled test input features.

        Returns
        -------
        X_test : ``numpy.ndarray`` or ``None``
            Scaled test input feature array, or None if not set.
        """
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value

    @property
    def y_test(self):
        """
        Test output labels (SNR values).

        Returns
        -------
        y_test : ``numpy.ndarray`` or ``None``
            Test output SNR values array, or None if not set.
        """
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    @property
    def error_adjustment(self):
        """
        Error correction parameters.

        Returns
        -------
        error_adjustment : ``dict`` or ``None``
            Dictionary with 'slope' and 'intercept' keys for linear error correction,
            or None if not computed.
        """
        return self._error_adjustment

    @error_adjustment.setter
    def error_adjustment(self, value):
        self._error_adjustment = value

    # -------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------
    def _get_input_output_data(self, params=None, randomize=True):
        """
        Generate input and output data arrays for ANN training.

        Extracts GW parameters, optionally randomizes the order, and prepares
        input feature arrays and corresponding SNR output values.

        Parameters
        ----------
        params : ``dict`` or ``str``
            GW parameter dictionary or path to JSON file containing parameters.
            Must include keys: 'mass_1', 'mass_2', 'luminosity_distance', etc. Optional.
        randomize : ``bool``
            If True, randomly shuffle the parameter order.\n
            default: True

        Returns
        -------
        X1 : ``numpy.ndarray``
            Input feature array of shape (N, 5) with columns:
            [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn].
        y1 : ``numpy.ndarray``
            Output SNR values array of shape (N,).

        Raises
        ------
        ValueError
            If output SNR data ('optimal_snr_net' or detector key) not found in params.

        Notes
        -----
        - Uses first detector in :attr:`~gwsnr.detector_list` for output SNR \n
        - Calls :meth:`~_get_input_data` for feature extraction \n
        """
        params = self._get_parameters(params)

        # Randomize parameter order if requested
        if randomize:
            idx = np.random.choice(len(params['mass_1']), size=len(params['mass_1']), replace=False)
        else:
            idx = np.arange(len(params['mass_1']))

        for key, value in params.items():
            params[key] = np.array(value)[idx]

        # Generate input features
        X1 = self._get_input_data(params=params)

        # Extract output SNR values
        det_ = self.gwsnr.detector_list[0]
        if 'optimal_snr_net' in params:
            y1 = np.array(params['optimal_snr_net'])
        elif det_ in params:
            y1 = np.array(params[det_])
        else:
            raise ValueError("Output data (snr) not found")

        return X1, y1
        
    def _get_input_data(self, params):
        """
        Extract ANN input features from gravitational wave parameters.

        Computes input feature array for neural network prediction using
        interpolated partial-scaled SNR and derived intrinsic parameters.

        Parameters
        ----------
        params : ``dict``
            GW parameter dictionary with required keys: \n
            - 'mass_1', 'mass_2': Component masses in solar masses \n
            - 'luminosity_distance': Distance in Mpc \n
            - 'theta_jn': Inclination angle in radians \n
            - 'psi', 'geocent_time', 'ra', 'dec': Extrinsic parameters \n
            - 'a_1', 'a_2', 'tilt_1', 'tilt_2': Spin parameters \n

        Returns
        -------
        X1 : ``numpy.ndarray``
            Input feature array of shape (N, 5) with columns:
            [partial_scaled_snr, amplitude_factor, eta, chi_eff, theta_jn].

        Raises
        ------
        ValueError
            If more than one detector is configured in GWSNR instance.

        Notes
        -----
        - Computes effective spin: chi_eff = (m1*a1z + m2*a2z) / (m1+m2) \n
        - Computes symmetric mass ratio: eta = m1*m2 / (m1+m2)^2 \n
        - Amplitude factor: A1 = Mc^(5/6) / d_eff \n
        - Uses aligned spin components: a_i * cos(tilt_i) \n
        """
        # Extract parameters from dictionary
        mass_1 = np.array(params['mass_1'])
        mass_2 = np.array(params['mass_2'])
        luminosity_distance = np.array(params['luminosity_distance'])
        theta_jn = np.array(params['theta_jn'])
        psi = np.array(params['psi'])
        geocent_time = np.array(params['geocent_time'])
        ra = np.array(params['ra'])
        dec = np.array(params['dec'])
        a_1 = np.array(params['a_1'])
        a_2 = np.array(params['a_2'])
        tilt_1 = np.array(params['tilt_1'])
        tilt_2 = np.array(params['tilt_2'])

        # Compute effective spin
        chi_eff = (mass_1 * a_1 * np.cos(tilt_1) + mass_2 * a_2 * np.cos(tilt_2)) / (mass_1 + mass_2)

        # Convert to aligned spin components
        a_1 = a_1 * np.cos(tilt_1)
        a_2 = a_2 * np.cos(tilt_2)
        
        # Validate single detector configuration
        detector_tensor = np.array(self.gwsnr.detector_tensor_list)
        len_ = len(detector_tensor)
        if len_ != 1:
            raise ValueError("Only one detector is allowed")
        
        # Calculate partial-scaled SNR using interpolation
        _, _, snr_partial_, d_eff = self.gwsnr.get_interpolated_snr(
            mass_1 = np.array(mass_1),
            mass_2 = np.array(mass_2),
            luminosity_distance = np.array(luminosity_distance),
            theta_jn = np.array(theta_jn),
            psi = np.array(psi),
            geocent_time = np.array(geocent_time),
            ra = np.array(ra),
            dec = np.array(dec),
            a_1 = np.array(a_1),
            a_2 = np.array(a_2),
            detector_tensor = detector_tensor,
            snr_partialscaled = np.array(self.gwsnr.snr_partialsacaled_list),
            ratio_arr = np.array(self.gwsnr.ratio_arr),
            mtot_arr = np.array(self.gwsnr.mtot_arr),
            a1_arr = np.array(self.gwsnr.a_1_arr),
            a_2_arr = np.array(self.gwsnr.a_2_arr),
        )

        # Compute derived quantities
        Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
        eta = mass_1 * mass_2/(mass_1 + mass_2)**2.
        amp0 = Mc ** (5.0 / 6.0) / np.array(d_eff)[0]

        # Construct input feature array
        X1 = np.vstack([np.array(snr_partial_)[0], amp0, eta, chi_eff, theta_jn]).T

        return X1

    def _standard_scaling_initialization(self, X_train):
        """
        Initialize and fit StandardScaler on training data.

        Creates a StandardScaler instance, fits it to the training data,
        and stores it as an instance attribute for later use.

        Parameters
        ----------
        X_train : ``numpy.ndarray``
            Training input features of shape (N, M).

        Returns
        -------
        X_train : ``numpy.ndarray``
            Scaled training features of shape (N, M).
        sc : ``sklearn.preprocessing.StandardScaler``
            Fitted StandardScaler instance.

        Notes
        -----
        - Stores fitted scaler in :attr:`~scaler` for prediction use \n
        - StandardScaler: z = (x - mean) / std \n
        """
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        self.scaler = sc

        return X_train, sc

    def _ann_model_4layers(self, 
        num_nodes_list,
        activation_fn_list,
        optimizer,
        loss,
        metrics,
    ):
        """
        Create a multi-layer sequential ANN model.

        Constructs a TensorFlow/Keras Sequential model with configurable
        layer sizes and activation functions.

        Parameters
        ----------
        num_nodes_list : ``list`` of ``int``
            Number of nodes in each layer. First element is input size,
            last element is output size. Example: [5, 32, 32, 1].
        activation_fn_list : ``list`` of ``str``
            Activation function for each layer. Must match length of num_nodes_list.
            Example: ['relu', 'relu', 'sigmoid', 'linear'].
        optimizer : ``str``
            Keras optimizer name (e.g., 'adam', 'sgd').
        loss : ``str``
            Keras loss function name (e.g., 'mean_squared_error', 'binary_crossentropy').
        metrics : ``list`` of ``str``
            List of metrics to evaluate during training (e.g., ['accuracy']).

        Returns
        -------
        ann : ``tensorflow.keras.Sequential``
            Compiled Keras Sequential model ready for training.

        Notes
        -----
        - Creates Dense layers starting from index 1 of num_nodes_list \n
        - First layer receives input implicitly based on training data shape \n
        """
        ann = tf.keras.models.Sequential()

        for i in range(1, len(num_nodes_list)):
            ann.add(tf.keras.layers.Dense(units=num_nodes_list[i], activation=activation_fn_list[i]))

        # Compile the ANN
        ann.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return ann

    def _get_parameters(self, gw_param_dict):
        """
        Load GW parameters from dictionary or JSON file.

        Handles parameter input as either a dictionary or a path to a JSON file,
        returning a standardized parameter dictionary.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or path to JSON file.

        Returns
        -------
        gw_param_dict : ``dict``
            Loaded parameter dictionary.

        Raises
        ------
        ValueError
            If gw_param_dict is neither a dictionary nor a valid file path.
        """
        if isinstance(gw_param_dict, str):
            path_ = f"{gw_param_dict}"
            gw_param_dict = get_param_from_json(path_)
        elif isinstance(gw_param_dict, dict):
            pass
        else:
            raise ValueError("gw_param_dict must be a dictionary or a json file")

        return gw_param_dict

    def _get_scaled_data(self, gw_param_dict, randomize=True, test_size=0.1, random_state=42):
        """
        Get scaled input/output data for prediction.

        Loads parameters, generates input features, and applies the fitted
        StandardScaler transformation.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or path to JSON file.
        randomize : ``bool``
            If True, randomly shuffle the parameter order.\n
            default: True
        test_size : ``float``
            Fraction of data to use for testing (unused in current implementation).\n
            default: 0.1
        random_state : ``int``
            Random state for reproducibility (unused in current implementation).\n
            default: 42

        Returns
        -------
        X : ``numpy.ndarray``
            Scaled input feature array.
        y : ``numpy.ndarray``
            Output SNR values.

        Notes
        -----
        - Requires :attr:`~scaler` to be fitted via :meth:`~_standard_scaling_initialization` \n
        """
        gw_param_dict = self._get_parameters(gw_param_dict)

        # Generate input and output data
        X, y = self._get_input_output_data(params=gw_param_dict, randomize=randomize)

        # Apply scaling
        X = self.scaler.transform(X)

        return X, y

    def _save_ann_path_dict(self, 
        ann_file_name='ann_model.h5', 
        scaler_file_name='scaler.pkl', 
        error_adjustment_file_name='error_adjustment.json',
        ann_path_dict_file_name='ann_path_dict.json',
    ):
        """
        Save ANN model configuration paths to JSON file.

        Creates or updates a configuration dictionary with paths to model files
        and associated parameters for the current detector.

        Parameters
        ----------
        ann_file_name : ``str``
            Filename of the trained model.\n
            default: 'ann_model.h5'
        scaler_file_name : ``str``
            Filename of the fitted scaler.\n
            default: 'scaler.pkl'
        error_adjustment_file_name : ``str``
            Filename of error adjustment parameters.\n
            default: 'error_adjustment.json'
        ann_path_dict_file_name : ``str``
            Output filename for the path configuration dictionary.\n
            default: 'ann_path_dict.json'

        Returns
        -------
        ann_path_dict : ``dict``
            Updated path configuration dictionary.

        Raises
        ------
        ValueError
            If any of the model, scaler, or error adjustment files don't exist.

        Notes
        -----
        - Saves to :attr:`~directory`/{ann_path_dict_file_name} \n
        - Includes sampling_frequency, minimum_frequency, waveform_approximant, snr_th \n
        """
        # Load existing configuration or create new
        if not os.path.exists(f'{self.directory}/{ann_path_dict_file_name}'):
            ann_path_dict = {}
        else:
            ann_path_dict = load_json(f'{self.directory}/{ann_path_dict_file_name}')

        # Validate file existence
        if not os.path.exists(f'{self.directory}/{ann_file_name}'):
            raise ValueError("Model file does not exist")
        if not os.path.exists(f'{self.directory}/{scaler_file_name}'):
            raise ValueError("Scaler file does not exist")
        if not os.path.exists(f'{self.directory}/{error_adjustment_file_name}'):
            raise ValueError("Error adjustment file does not exist")
            
        # Build configuration entry for current detector
        ann_path_dict_ = {
            self.gwsnr.detector_list[0]: {
                "model_path": f'{self.directory}/{ann_file_name}',
                "scaler_path": f'{self.directory}/{scaler_file_name}',
                "error_adjustment_path": f'{self.directory}/{error_adjustment_file_name}',
                "sampling_frequency": self.gwsnr_args['sampling_frequency'],
                "minimum_frequency": self.gwsnr_args['minimum_frequency'],
                "waveform_approximant": self.gwsnr_args['waveform_approximant'], 
                "snr_th": self.gwsnr_args['snr_th'],},
        }

        # Update and save configuration
        ann_path_dict.update(ann_path_dict_)
        append_json(f'{self.directory}/{ann_path_dict_file_name}', ann_path_dict, replace=True)
        print(f"ann path dict saved at: {self.directory}/{ann_path_dict_file_name}")

        return ann_path_dict

    def _helper_error_adjustment(self, y_pred, y_test, snr_range=[4,10]):
        """
        Calculate linear error adjustment parameters via curve fitting.

        Fits a linear model to prediction errors within a specified SNR range
        to improve prediction accuracy.

        Parameters
        ----------
        y_pred : ``numpy.ndarray``
            Predicted SNR values.
        y_test : ``numpy.ndarray``
            True SNR values.
        snr_range : ``list`` of ``float``
            SNR range [min, max] for fitting. Only samples within this range are used.\n
            default: [4, 10]

        Returns
        -------
        adjustment_dict : ``dict``
            Dictionary with keys 'slope' and 'intercept' for linear correction:
            error = slope * y_pred + intercept.

        Notes
        -----
        - Error correction: y_adjusted = y_pred - (slope * y_pred + intercept) \n
        - Uses scipy.optimize.curve_fit for linear regression \n
        - Excludes samples where y_test == 0 \n
        """
        def linear_fit_fn(x, a, b):
            return a*x + b

        # Select samples within SNR range and non-zero true values
        idx = (y_pred>snr_range[0]) & (y_pred<snr_range[1])
        idx &= (y_test != 0)

        # Fit linear model to errors
        popt, pcov = curve_fit(linear_fit_fn, y_pred[idx], y_pred[idx]-y_test[idx])

        adjustment_dict = {'slope': popt[0], 'intercept': popt[1]}

        return adjustment_dict

    # -------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------
    def ann_model_training(self,
        gw_param_dict,
        randomize=True,
        test_size=0.1,
        random_state=42,
        num_nodes_list = [5, 32, 32, 1],
        activation_fn_list = ['relu', 'relu', 'sigmoid', 'linear'],
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy'],
        batch_size=32,
        epochs=100,
        error_adjustment_snr_range=[4,10],
        ann_file_name = 'ann_model.h5',
        scaler_file_name = 'scaler.pkl',
        error_adjustment_file_name='error_adjustment.json',
        ann_path_dict_file_name='ann_path_dict.json',
    ):
        """
        Train ANN model for SNR prediction using GW parameter data.

        Complete training pipeline including data preparation, model training,
        error adjustment calculation, and file saving.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or path to JSON file containing training data.
        randomize : ``bool``
            If True, randomly shuffle the training data.\n
            default: True
        test_size : ``float``
            Fraction of data to hold out for testing.\n
            default: 0.1
        random_state : ``int``
            Random state for train/test split reproducibility.\n
            default: 42
        num_nodes_list : ``list`` of ``int``
            Number of nodes in each layer.\n
            default: [5, 32, 32, 1]
        activation_fn_list : ``list`` of ``str``
            Activation functions for each layer.\n
            default: ['relu', 'relu', 'sigmoid', 'linear']
        optimizer : ``str``
            Keras optimizer name.\n
            default: 'adam'
        loss : ``str``
            Keras loss function name.\n
            default: 'mean_squared_error'
        metrics : ``list`` of ``str``
            Metrics to evaluate during training.\n
            default: ['accuracy']
        batch_size : ``int``
            Batch size for training.\n
            default: 32
        epochs : ``int``
            Number of training epochs.\n
            default: 100
        error_adjustment_snr_range : ``list`` of ``float``
            SNR range [min, max] for computing error adjustment parameters.\n
            default: [4, 10]
        ann_file_name : ``str``
            Output filename for trained model.\n
            default: 'ann_model.h5'
        scaler_file_name : ``str``
            Output filename for fitted scaler.\n
            default: 'scaler.pkl'
        error_adjustment_file_name : ``str``
            Output filename for error adjustment parameters.\n
            default: 'error_adjustment.json'
        ann_path_dict_file_name : ``str``
            Output filename for ANN configuration paths.\n
            default: 'ann_path_dict.json'

        Notes
        -----
        - Saves model to :attr:`~directory`/{ann_file_name} \n
        - Saves scaler to :attr:`~directory`/{scaler_file_name} \n
        - Computes error adjustment using :meth:`~_helper_error_adjustment` \n
        - Stores test data in :attr:`~X_test` and :attr:`~y_test` \n

        Examples
        --------
        >>> amg = ANNModelGenerator()
        >>> amg.ann_model_training(
                gw_param_dict='training_params.json',
                epochs=200,
                batch_size=64)
        """
        # Generate input and output data
        X, y = self._get_input_output_data(params=gw_param_dict, randomize=randomize)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Initialize and fit scaler
        X_scaled, scaler = self._standard_scaling_initialization(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_test = y_test

        # Save the fitted scaler
        pickle.dump(scaler, open(f'{self.directory}/{scaler_file_name}', 'wb'))

        # Initialize and train the ANN model
        ann = self.ann_model(num_nodes_list, activation_fn_list, optimizer, loss, metrics)
        ann.fit(X_scaled, y_train, batch_size=batch_size, epochs=epochs)
        self.ann = ann

        # Save the trained model
        ann.save(f'{self.directory}/{ann_file_name}')

        # Compute and save error adjustment parameters
        y_pred = np.array(ann.predict(self.X_test)).flatten()
        self.error_adjustment = self._helper_error_adjustment(y_pred, y_test, snr_range=error_adjustment_snr_range)
        append_json(f'{self.directory}/{error_adjustment_file_name}', self.error_adjustment, replace=True)

        # Print save locations
        print(f"scaler saved at: {self.directory}/{scaler_file_name}")
        print(f"model saved at: {self.directory}/{ann_file_name}")
        print(f"error adjustment saved at: {self.directory}/{error_adjustment_file_name}")

        # Save ANN path configuration
        self._save_ann_path_dict(
            ann_file_name=ann_file_name, 
            scaler_file_name=scaler_file_name, 
            error_adjustment_file_name=error_adjustment_file_name,
            ann_path_dict_file_name=ann_path_dict_file_name,
        )

    def pdet_error(self, gw_param_dict=None, randomize=True, error_adjustment=True):
        """
        Calculate detection probability error rate.

        Evaluates the percentage of samples where predicted and true detection
        status (SNR > threshold) differ.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or JSON file path. If None, uses stored test data. Optional.
        randomize : ``bool``
            If True, randomly shuffle parameters (only used if gw_param_dict provided).\n
            default: True
        error_adjustment : ``bool``
            If True, apply linear error correction to predictions.\n
            default: True

        Returns
        -------
        error : ``float``
            Percentage of misclassified samples.
        y_pred : ``numpy.ndarray``
            Predicted SNR values (with or without error adjustment).
        y_test : ``numpy.ndarray``
            True SNR values.

        Notes
        -----
        - Uses :attr:`~gwsnr_args['snr_th']` as detection threshold \n
        - Error adjustment: y_adj = y_pred - (slope*y_pred + intercept) \n
        """
        snr_threshold = self.gwsnr_args['snr_th']

        # Load or use stored test data
        if gw_param_dict is not None:
            gw_param_dict = self._get_parameters(gw_param_dict)
            X_test, y_test = self._get_input_output_data(params=gw_param_dict, randomize=randomize)
            X_test = self.scaler.transform(X_test)
        else:
            X_test = self.X_test
            y_test = self.y_test

        # Predict SNR values
        y_pred = self.ann.predict(X_test).flatten()

        # Apply error adjustment if requested
        if error_adjustment:
            adjustment_dict = self.error_adjustment
            a = adjustment_dict['slope']
            b = adjustment_dict['intercept']
            y_pred = y_pred-(a*y_pred + b)

        # Calculate error rate
        len1 = len(y_pred)
        len2 = np.sum((y_pred>snr_threshold) != (y_test>snr_threshold))
        error = len2/len1*100
        print(f"Error: {error:.2f}%")

        return error, y_pred, y_test

    def pdet_confusion_matrix(self, gw_param_dict=None, randomize=True, snr_threshold=8.0):
        """
        Generate confusion matrix for detection probability classification.

        Evaluates ANN predictions as binary classification (detected/not detected)
        and computes confusion matrix and accuracy metrics.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or JSON file path. If None, uses stored test data. Optional.
        randomize : ``bool``
            If True, randomly shuffle parameters (only used if gw_param_dict provided).\n
            default: True
        snr_threshold : ``float``
            SNR threshold for detection classification.\n
            default: 8.0

        Returns
        -------
        cm : ``numpy.ndarray``
            Confusion matrix of shape (2, 2).
        accuracy : ``float``
            Classification accuracy percentage.
        y_pred : ``numpy.ndarray``
            Predicted detection status (boolean array).
        y_test : ``numpy.ndarray``
            True detection status (boolean array).

        Notes
        -----
        - Uses sklearn.metrics.confusion_matrix and accuracy_score \n
        - Prints confusion matrix and accuracy to stdout \n
        """
        # Load or use stored test data
        if gw_param_dict is not None:
            gw_param_dict = self._get_parameters(gw_param_dict)
            X_test, y_test = self._get_input_output_data(params=gw_param_dict, randomize=randomize)
            X_test = self.scaler.transform(X_test)
        else:
            X_test = self.X_test
            y_test = self.y_test

        # Predict and apply threshold
        y_pred = self.ann.predict(X_test).flatten()
        y_pred = (y_pred>snr_threshold)
        y_test = (y_test>snr_threshold)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)*100
        print(f"Accuracy: {accuracy:.3f}%")

        return cm, accuracy, y_pred, y_test

    def load_model_scaler_error(self, 
        ann_file_name='ann_model.h5',
        scaler_file_name='scaler.pkl',
        error_adjustment_file_name=False,
    ):
        """
        Load pre-trained ANN model, scaler, and optionally error adjustment.

        Restores saved model components for prediction use.

        Parameters
        ----------
        ann_file_name : ``str``
            Filename of the trained model.\n
            default: 'ann_model.h5'
        scaler_file_name : ``str``
            Filename of the fitted scaler.\n
            default: 'scaler.pkl'
        error_adjustment_file_name : ``str`` or ``bool``
            Filename of error adjustment parameters. If False, not loaded.\n
            default: False

        Returns
        -------
        ann : ``tensorflow.keras.Model``
            Loaded Keras model.
        scaler : ``sklearn.preprocessing.StandardScaler``
            Loaded scaler.
        error_adjustment : ``dict``
            Error adjustment parameters (only returned if error_adjustment_file_name is provided). Optional.

        Notes
        -----
        - Updates :attr:`~ann`, :attr:`~scaler`, and optionally :attr:`~error_adjustment` \n
        - Files are loaded from :attr:`~directory` \n
        """
        # Load model and scaler
        self.ann = load_ann_h5(f'{self.directory}/{ann_file_name}')
        self.scaler = pickle.load(open(f'{self.directory}/{scaler_file_name}', 'rb'))

        # Optionally load error adjustment
        if error_adjustment_file_name:
            self.error_adjustment = get_param_from_json(f'{self.directory}/{error_adjustment_file_name}')
            return self.ann, self.scaler, self.error_adjustment
        else:
            return self.ann, self.scaler

    def snr_error_adjustment(self, gw_param_dict=None, randomize=True, snr_range=[4,10], error_adjustment_file_name='error_adjustment.json'):
        """
        Recalculate and save error adjustment parameters.

        Computes new error adjustment based on current predictions and updates
        the stored parameters.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or JSON file path for evaluation. Optional.
        randomize : ``bool``
            If True, randomly shuffle parameters.\n
            default: True
        snr_range : ``list`` of ``float``
            SNR range for error adjustment fitting.\n
            default: [4, 10]
        error_adjustment_file_name : ``str``
            Output filename for updated error adjustment.\n
            default: 'error_adjustment.json'

        Returns
        -------
        error_adjustment : ``dict``
            Updated error adjustment parameters with 'slope' and 'intercept'.

        Notes
        -----
        - Calls :meth:`~pdet_error` with error_adjustment=True for predictions \n
        - Saves updated parameters to :attr:`~directory`/{error_adjustment_file_name} \n
        """

        # Get predictions
        _, y_pred_, y_test_ = self.pdet_error(gw_param_dict=gw_param_dict, randomize=True, error_adjustment=True)

        # Calculate new error adjustment
        self.error_adjustment = self._helper_error_adjustment(y_pred_, y_test_, snr_range=snr_range)

        print(f"slope: {self.error_adjustment['slope']:.4f}, intercept: {self.error_adjustment['intercept']:.4f}")

        # Save to file
        append_json(f'{self.directory}/{error_adjustment_file_name}', self.error_adjustment, replace=True)
        print(f"error adjustment saved at: {self.directory}/{error_adjustment_file_name}")

        return self.error_adjustment

    def predict_snr(self, gw_param_dict, error_adjustment=True):
        """
        Predict SNR values using trained ANN model.

        Applies the trained model to new GW parameters and optionally
        applies error correction.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or path to JSON file.
        error_adjustment : ``bool``
            If True, apply linear error correction to predictions.\n
            default: True

        Returns
        -------
        y_pred : ``numpy.ndarray``
            Predicted SNR values (corrected if error_adjustment=True).

        Notes
        -----
        - Requires :attr:`~ann` and :attr:`~scaler` to be loaded/trained \n
        - Error adjustment: y_adj = y_pred - (slope*y_pred + intercept) \n
        """
        params = self._get_parameters(gw_param_dict)

        # Generate input features
        X_test = self._get_input_data(params=params)
        X_test = self.scaler.transform(X_test)

        # Predict SNR
        y_pred = self.ann.predict(X_test).flatten()

        # Apply error adjustment if requested
        if error_adjustment:
            adjustment_dict = self.error_adjustment
            a = adjustment_dict['slope']
            b = adjustment_dict['intercept']
            y_pred = y_pred-(a*y_pred + b)

        return y_pred

    def predict_pdet(self, gw_param_dict, snr_threshold=8.0, error_adjustment=True):
        """
        Predict detection probability using trained ANN model.

        Classifies events as detected (SNR > threshold) or not detected
        based on ANN predictions.

        Parameters
        ----------
        gw_param_dict : ``dict`` or ``str``
            GW parameter dictionary or path to JSON file.
        snr_threshold : ``float``
            SNR threshold for detection classification.\n
            default: 8.0
        error_adjustment : ``bool``
            If True, apply error correction before thresholding.\n
            default: True

        Returns
        -------
        y_pred : ``numpy.ndarray`` of ``bool``
            Detection status for each sample (True = detected).

        Notes
        -----
        - Calls :meth:`~predict_snr` for SNR prediction \n
        - Returns boolean array: y_pred > snr_threshold \n
        """
        y_pred = self.predict_snr(gw_param_dict, error_adjustment=error_adjustment)
        y_pred = (y_pred>snr_threshold)

        return y_pred