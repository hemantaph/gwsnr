# -*- coding: utf-8 -*-
"""
This module contains the ANNModelGenerator class which is used to generate the ANN (Artificial Neural Network) model that can be used to predict the SNR of the GW events.
"""

import numpy as np
import os
import pickle
from .gwsnr import GWSNR
from .utils import append_json, get_param_from_json, load_json, load_pickle, save_pickle, load_ann_h5
from scipy.optimize import curve_fit

from gwsnr import antenna_response_array, cubic_spline_interpolator2d
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class ANNModelGenerator():
    """
    ANNModelGenerator class is used to generate the ANN model that can be used to predict the SNR of the GW events.

    Parameters
    ----------
    npool : `int`
        Number of processors to use for parallel processing.
        Default is 4.
    gwsnr_verbose : `bool`
        If True, print the progress of the GWSNR calculation.
        Default is True.
    snr_th : `float`
        SNR threshold for the error calculation.
        Default is 8.0.
    waveform_approximant : `str`
        Waveform approximant to be used for the GWSNR calculation and the ANN model.
        Default is "IMRPhenomXPHM".
    **kwargs : `dict`
        Keyword arguments for the GWSNR class.
        To see the list of available arguments, 
        >>> from gwsnr import GWSNR
        >>> help(GWSNR)
        
    Examples
    --------
    >>> from gwsnr import ANNModelGenerator
    >>> amg = ANNModelGenerator()
    >>> amg.ann_model_training(gw_param_dict='gw_param_dict.json') # training the ANN model with pre-generated parameter points

    """
    
    def __init__(self,
        directory="./gwsnr_data",
        npool=4,
        gwsnr_verbose=True,
        snr_th=8.0,
        waveform_approximant="IMRPhenomXPHM",
        **kwargs,  # ler and gwsnr arguments
    ):

        self.directory = directory
        self.ann_model = self.ann_model_4layers
        self.ann = None
        self.scaler = None

        self.gwsnr_args = dict(
            npool=npool,
            # gwsnr args
            mtot_min=2.0,
            mtot_max=200,
            ratio_min=0.1,
            ratio_max=1.0,
            mtot_resolution=500,
            ratio_resolution=50,
            sampling_frequency=2048.0,
            waveform_approximant=waveform_approximant,
            minimum_frequency=20.0,
            snr_type="interpolation",
            psds=None,
            ifos=None,
            interpolator_dir="./interpolator_pickle",
            create_new_interpolator=False,
            gwsnr_verbose=gwsnr_verbose,
            multiprocessing_verbose=True,
            mtot_cut=True,
            snr_th=snr_th,
        )
        self.gwsnr_args.update(kwargs)

        # gwsnr initialization
        # spinless
        self.gwsnr = GWSNR(
            npool=self.gwsnr_args['npool'],
            # gwsnr args
            mtot_min=self.gwsnr_args['mtot_min'],
            mtot_max=self.gwsnr_args['mtot_max'],
            ratio_min=self.gwsnr_args['ratio_min'],
            ratio_max=self.gwsnr_args['ratio_max'],
            mtot_resolution=self.gwsnr_args['mtot_resolution'],
            ratio_resolution=self.gwsnr_args['ratio_resolution'],
            sampling_frequency=self.gwsnr_args['sampling_frequency'],
            waveform_approximant=self.gwsnr_args['waveform_approximant'],
            minimum_frequency=self.gwsnr_args['minimum_frequency'],
            snr_type='interpolation',
            psds=self.gwsnr_args['psds'],
            ifos=self.gwsnr_args['ifos'],
            interpolator_dir=self.gwsnr_args['interpolator_dir'],
            create_new_interpolator=self.gwsnr_args['create_new_interpolator'],
            gwsnr_verbose=self.gwsnr_args['gwsnr_verbose'],
            multiprocessing_verbose=self.gwsnr_args['multiprocessing_verbose'],
            mtot_cut=self.gwsnr_args['mtot_cut'],
            snr_th=self.gwsnr_args['snr_th'],
        )

    def get_input_output_data(self, params=None, randomize=True):

        params = self.get_parameters(params)

        if randomize:
            idx = np.random.choice(len(params['mass_1']), size=len(params['mass_1']), replace=False)
        else:
            idx = np.arange(len(params['mass_1']))

        for key, value in params.items():
            params[key] = np.array(value)[idx]

        X1 = self.get_input_data(params=params)
        # output data
        # get snr for y train
        det_ = self.gwsnr.detector_list[0]
        if 'optimal_snr_net' in params:
            y1 = np.array(params['optimal_snr_net'])
        elif det_ in params:
            y1 = np.array(params[det_])
        else:
            raise ValueError("Output data (snr) not found")

        return X1, y1
        
    def get_input_data(self, params):
        """
        Function to generate input and output data for the neural network

        Parameters:
        idx: index of the parameter points
        params: dictionary of parameter points
            params.keys() = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'psi', 'geocent_time', 'ra', 'dec', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'L1']

        Returns:
        X: input data, [snr_partial_[0], amp0[0], eta, chi_eff, theta_jn]
        y: output data, [L1]
        """

        mass_1 = np.array(params['mass_1'])
        mass_2 = np.array(params['mass_2'])
        luminosity_distance = np.array(params['luminosity_distance'])
        theta_jn = np.array(params['theta_jn'])
        psi = np.array(params['psi'])
        geocent_time = np.array(params['geocent_time'])
        ra = np.array(params['ra'])
        dec = np.array(params['dec'])
        
        detector_tensor = self.gwsnr.detector_tensor_list
        snr_partial_coeff = np.array(self.gwsnr.snr_partialsacaled_list)[0]
        ratio_arr = self.gwsnr.ratio_arr
        mtot_arr = self.gwsnr.mtot_arr
        
        size = len(mass_1)
        len_ = len(detector_tensor)
        if len_ != 1:
            raise ValueError("Only one detector is allowed")

        mtot = mass_1 + mass_2
        ratio = mass_2 / mass_1
        # get array of antenna response
        Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)
        Fp = np.array(Fp[0])
        Fc = np.array(Fc[0])

        Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
        eta = mass_1 * mass_2/(mass_1 + mass_2)**2.
        A1 = Mc ** (5.0 / 6.0)
        ci_2 = np.cos(theta_jn) ** 2
        ci_param = ((1 + np.cos(theta_jn) ** 2) / 2) ** 2
        
        size = len(mass_1)
        snr_partial_ = np.zeros(size)
        d_eff = np.zeros(size)

        # loop over the detectors
        for i in range(size):
            snr_partial_[i] = cubic_spline_interpolator2d(mtot[i], ratio[i], snr_partial_coeff, mtot_arr, ratio_arr)
            d_eff[i] =luminosity_distance[i] / np.sqrt(
                    Fp[i]**2 * ci_param[i] + Fc[i]**2 * ci_2[i]
                )
        #amp0
        amp0 =  A1 / d_eff

        # get spin parameters
        a_1 = np.array(params['a_1'])
        a_2 = np.array(params['a_2'])
        tilt_1 = np.array(params['tilt_1'])
        tilt_2 = np.array(params['tilt_2'])

        # effective spin
        chi_eff = (mass_1 * a_1 * np.cos(tilt_1) + mass_2 * a_2 * np.cos(tilt_2)) / (mass_1 + mass_2)


        # input data
        X1 = np.vstack([snr_partial_, amp0, eta, chi_eff, theta_jn]).T

        return X1

    def standard_scaling_initialization(self, X_train):

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        self.scaler = sc

        return X_train, sc

    def ann_model_4layers(self, 
        num_nodes_list,
        activation_fn_list,
        optimizer,
        loss,
        metrics,
    ):
        ann = tf.keras.models.Sequential()

        for i in range(1, len(num_nodes_list)):
            ann.add(tf.keras.layers.Dense(units=num_nodes_list[i], activation=activation_fn_list[i]))

        # compile the ANN
        ann.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return ann

    def get_parameters(self, gw_param_dict):

        # get the parameters
        if isinstance(gw_param_dict, str):
            path_ = f"{self.directory}/{gw_param_dict}"
            gw_param_dict = get_param_from_json(path_)
        elif isinstance(gw_param_dict, dict):
            pass
        else:
            raise ValueError("gw_param_dict must be a dictionary or a json file")

        return gw_param_dict

    def get_scaled_data(self, gw_param_dict, randomize=True, test_size=0.1, random_state=42):

        gw_param_dict = self.get_parameters(gw_param_dict)

        # input and output data
        X, y = self.get_input_output_data(params=gw_param_dict, randomize=randomize)

        # scaling
        X = self.scaler.transform(X)

        return X, y

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
        snr_threshold = self.gwsnr_args['snr_th']   # snr threshold

        # input and output data
        X, y = self.get_input_output_data(params=gw_param_dict, randomize=randomize)

        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # scaling
        X_scaled, scaler = self.standard_scaling_initialization(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_test = y_test
        # save the scaler
        pickle.dump(scaler, open(f'{self.directory}/{scaler_file_name}', 'wb'))

        # initialize the ANN
        ann = self.ann_model(num_nodes_list, activation_fn_list, optimizer, loss, metrics)

        # fit the ANN to the training set
        ann.fit(X_scaled, y_train, batch_size=batch_size, epochs=epochs)
        # for testing
        self.ann = ann
        # save the model
        ann.save(f'{self.directory}/{ann_file_name}')

        # error adjustment
        y_pred = np.array(ann.predict(self.X_test)).flatten()
        self.error_adjustment = self.helper_error_adjustment(y_pred, y_test, snr_range=error_adjustment_snr_range)
        append_json(f'{self.directory}/{error_adjustment_file_name}', self.error_adjustment, replace=True)

        # print the results
        print(f"scaler saved at: {self.directory}/{scaler_file_name}")
        print(f"model saved at: {self.directory}/{ann_file_name}")
        print(f"error adjustment saved at: {self.directory}/{error_adjustment_file_name}")

        # save the path of the model, scaler, and error adjustment
        self.save_ann_path_dict(
            ann_file_name=ann_file_name, 
            scaler_file_name=scaler_file_name, 
            error_adjustment_file_name=error_adjustment_file_name,
            ann_path_dict_file_name=ann_path_dict_file_name,
        )

    def pdet_error(self, gw_param_dict=None, randomize=True, error_adjustment=True):

        snr_threshold = self.gwsnr_args['snr_th']   # snr threshold

        if gw_param_dict is not None:
            gw_param_dict = self.get_parameters(gw_param_dict)
            # input and output data
            X_test, y_test = self.get_input_output_data(params=gw_param_dict, randomize=randomize)
            X_test = self.scaler.transform(X_test)
        else:
            X_test = self.X_test
            y_test = self.y_test

        # calculate the error
        y_pred = self.ann.predict(X_test).flatten()

        # error adjustment
        if error_adjustment:
            adjustment_dict = self.error_adjustment
            a = adjustment_dict['slope']
            b = adjustment_dict['intercept']
            y_pred = y_pred-(a*y_pred + b)

        len1 = len(y_pred)
        len2 = np.sum((y_pred>snr_threshold) != (y_test>snr_threshold))
        error = len2/len1*100
        print(f"Error: {error:.2f}%")

        return error, y_pred, y_test

    def save_ann_path_dict(self, 
        ann_file_name='ann_model.h5', 
        scaler_file_name='scaler.pkl', error_adjustment_file_name='error_adjustment.json',
        ann_path_dict_file_name='ann_path_dict.json',
    ):

        if not os.path.exists(f'{self.directory}/{ann_path_dict_file_name}'):
            ann_path_dict = {}
        else:
            ann_path_dict = load_json(f'{self.directory}/{ann_path_dict_file_name}')

        # check if the files exist
        if not os.path.exists(f'{self.directory}/{ann_file_name}'):
            raise ValueError("Model file does not exist")
        if not os.path.exists(f'{self.directory}/{scaler_file_name}'):
            raise ValueError("Scaler file does not exist")
        if not os.path.exists(f'{self.directory}/{error_adjustment_file_name}'):
            raise ValueError("Error adjustment file does not exist")
            
        ann_path_dict_ = {
            "L1": {
                "model_path": f'{self.directory}/{ann_file_name}',
                "scaler_path": f'{self.directory}/{scaler_file_name}',
                "error_adjustment_path": f'{self.directory}/{error_adjustment_file_name}',
                "sampling_frequency": self.gwsnr_args['sampling_frequency'],
                "minimum_frequency": self.gwsnr_args['minimum_frequency'],
                "waveform_approximant": self.gwsnr_args['waveform_approximant'], 
                "snr_th": self.gwsnr_args['snr_th'],},
        }

        ann_path_dict.update(ann_path_dict_)
        append_json(f'{self.directory}/{ann_path_dict_file_name}', ann_path_dict, replace=True)
        print(f"ann path dict saved at: {self.directory}/{ann_path_dict_file_name}")

        return ann_path_dict

    def pdet_confusion_matrix(self, gw_param_dict=None, randomize=True, snr_threshold=8.0):

        if gw_param_dict is not None:
            gw_param_dict = self.get_parameters(gw_param_dict)
            # input and output data
            X_test, y_test = self.get_input_output_data(params=gw_param_dict, randomize=randomize)
            X_test = self.scaler.transform(X_test)
        else:
            X_test = self.X_test
            y_test = self.y_test

        # calculate the error
        y_pred = self.ann.predict(X_test).flatten()
        y_pred = (y_pred>snr_threshold)
        y_test = (y_test>snr_threshold)

        # # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        accuracy = accuracy_score(y_test, y_pred)*100
        print(f"Accuracy: {accuracy:.3f}%")

        return cm, accuracy, y_pred, y_test

    def load_model_scaler_error(self, 
        ann_file_name='ann_model.h5',
        scaler_file_name='scaler.pkl',
        error_adjustment_file_name=False,
    ):
        # load the model
        self.ann = load_ann_h5(f'{self.directory}/{ann_file_name}')
        self.scaler = pickle.load(open(f'{self.directory}/{scaler_file_name}', 'rb'))
        if error_adjustment_file_name:
            self.error_adjustment = get_param_from_json(f'{self.directory}/{error_adjustment_file_name}')

            return self.ann, self.scaler, self.error_adjustment
        else:
            return self.ann, self.scaler

    def helper_error_adjustment(self, y_pred, y_test, snr_range=[4,10]):

        def linear_fit_fn(x, a, b):
            return a*x + b

        idx = (y_pred>snr_range[0]) & (y_pred<snr_range[1])
        idx &= (y_test != 0)
        popt, pcov = curve_fit(linear_fit_fn, y_pred[idx], y_pred[idx]-y_test[idx])

        adjustment_dict = {'slope': popt[0], 'intercept': popt[1]}

        return adjustment_dict

    def snr_error_adjustment(self, gw_param_dict=None, randomize=True, snr_threshold=8.0, snr_range=[4,10], error_adjustment_file_name='error_adjustment.json'):

        _, y_pred_, y_test_ = self.pdet_error(gw_param_dict=gw_param_dict, randomize=True, snr_threshold=snr_threshold)

        self.error_adjustment = self.helper_error_adjustment(y_pred_, y_test_, snr_range=snr_range)

        print(f"slope: {self.error_adjustment['slope']:.4f}, intercept: {self.error_adjustment['intercept']:.4f}")

        # save json file
        append_json(f'{self.directory}/{error_adjustment_file_name}', self.error_adjustment, replace=True)
        print(f"error adjustment saved at: {self.directory}/{error_adjustment_file_name}")

        return self.error_adjustment

    def predict_snr(self, gw_param_dict, error_adjustment=True):

        params = self.get_parameters(gw_param_dict)

        # input and output data
        X_test = self.get_input_data(params=params)
        X_test = self.scaler.transform(X_test)

        # calculate the error
        y_pred = self.ann.predict(X_test).flatten()

        # error adjustment
        if error_adjustment:
            adjustment_dict = self.error_adjustment
            a = adjustment_dict['slope']
            b = adjustment_dict['intercept']
            y_pred = y_pred-(a*y_pred + b)

        return y_pred

    def predict_pdet(self, gw_param_dict, snr_threshold=8.0, error_adjustment=True):

        y_pred = self.predict_snr(gw_param_dict, error_adjustment=error_adjustment)
        y_pred = (y_pred>snr_threshold)

        return y_pred