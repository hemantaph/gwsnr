import multiprocessing as mp
# mp.set_start_method('fork', force=True)

from multiprocessing import Pool
import numpy as np
import jax
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", True)
from ripple import ms_to_Mc_eta
from jax import vmap
from jax import jit
from ..jax import findchirp_chirptime_jax

from tqdm import tqdm

from ..utils import noise_weighted_inner_prod_ripple

class RippleInnerProduct:
    """
    Class to compute the noise weighted inner product for a given waveform and PSD
    """

    def __init__(self, 
                 waveform_name='IMRPhenomD', 
                 minimum_frequency=20.0, 
                 sampling_frequency=2048.0, 
                 reference_frequency=None
                ):

        # instance initialization
        self.f_u = sampling_frequency/2
        self.f_ref = reference_frequency if reference_frequency is not None else minimum_frequency
        self.f_l = minimum_frequency

        # get list of arguments for the chosen waveform
        self.arg_list = self.arg_selection(waveform_name) 
        # get the instance to the waveform generator to compute h+ and hx
        # JAX function
        self.gen_hphc = jit(self.select_waveform(waveform_name))
        # use vmap instead of looping
        # JAX function
        self.vmap_waveform = vmap(self.gen_hphc, in_axes=(0, 0, None))
        # use vmap for findchirp chirp time
        # JAX function
        self.vmap_findchirp_chirptime = vmap(findchirp_chirptime_jax, in_axes=(0, 0, None))
        # # use vmap for inner product
        # self.vmap_noise_weighted_inner_product = jit(noise_weighted_inner_product)

        # self.vmap_arange = vmap(jnp.arange, in_axes=(None, 0, 0))

    # def noise_weighted_inner_product(
    #     self, signal1, signal2, psd, duration,
    # ):
    #     """
    #     Noise weighted inner product of two time series data sets.

    #     Parameters
    #     ----------
    #     signal1: `numpy.ndarray` or `float`
    #         First series data set.
    #     signal2: `numpy.ndarray` or `float`
    #         Second series data set.
    #     psd: `numpy.ndarray` or `float`
    #         Power spectral density of the detector.
    #     duration: `float`
    #         Duration of the data.
    #     """

    #     nwip_arr = np.conj(signal1) * signal2 / psd
    #     return 4 / duration * np.sum(nwip_arr)

    def arg_selection(self, waveform_name):
        """
        Returns the list of arguments required for the chosen waveform.

        Parameters
        ----------
        waveform_name: `str`
            The name of the waveform to use. Ripple supported waveforms only.
        
        Returns
        -------
        list: List of arguments required for the chosen waveform.
        """

        supported_waveforms = ['IMRPhenomXAS', 'IMRPhenomD', 'TaylorF2', 'IMRPhenomD_NRTidalv2']

        if waveform_name == 'IMRPhenomXAS':
            return ['Mchirp', 'eta', 'a_1', 'a_2', 'luminosity_distance', 'tc', 'phase', 'theta_jn']
        elif waveform_name == 'IMRPhenomD':
            return ['Mchirp', 'eta', 'a_1', 'a_2', 'luminosity_distance', 'tc', 'phase', 'theta_jn']
        elif waveform_name == 'TaylorF2':
            return ['Mchirp', 'eta', 'a_1', 'a_2', 'lambda_1', 'lambda_2','luminosity_distance', 'tc', 'phase', 'theta_jn']
        elif waveform_name == 'IMRPhenomD_NRTidalv2':
            return ['Mchirp', 'eta', 'a_1', 'a_2', 'lambda_1', 'lambda_2','luminosity_distance', 'tc', 'phase', 'theta_jn']
        else:
            raise ValueError(f"Waveform '{waveform_name}' not supported by ripple. Supported waveforms are: {supported_waveforms}")

    def select_waveform(self, waveform_name):
        """
        Imports and returns the specified waveform from ripple.waveforms.

        Parameters:
        waveform_name (str): The name of the waveform to import.

        Returns:
        class: The waveform class from ripple.waveforms.
        """

        try:
            # Import inside the function
            waveform_module = __import__('ripple.waveforms', fromlist=[waveform_name])
            mod_ = getattr(waveform_module, waveform_name)
            attr_ = "gen_"+waveform_name+"_hphc"
            return getattr(mod_, attr_)
        except AttributeError:
            raise ValueError(f"Waveform '{waveform_name}' not found in ripple.waveforms module.")

    # def waveform_polarization(self, mass_1, mass_2, luminosity_distance, theta_jn, phase, a_1, a_2):

    #     f_u = self.f_u
    #     f_l = self.f_l
    #     f_ref = f_l
    #     duration_max = 64.
    #     duration_min = 4.

    #     # set up waveform calculator inputs
    #     Mchirp, eta = jnp.array(ms_to_Mc_eta(jnp.array([mass_1, mass_2])), dtype=jnp.float32)
    #     tc = 0.
    #     # 'Mchirp', 'eta', 'a_1', 'a_2', 'luminosity_distance', 'tc', 'phase', 'theta_jn'
    #     theta_ripple = jnp.array([Mchirp, eta, a_1, a_2, luminosity_distance, tc, phase, theta_jn]).T

    #     safety = 1.1
    #     approx_duration = safety * self.vmap_findchirp_chirptime(mass_1, mass_2, f_l)  # vmap result give you a stack which is 2D JAX array
    #     duration = jnp.ceil(approx_duration + 2.0).flatten()  # coverts to numpy 1D array
    #     duration = jnp.clip(duration, duration_min, duration_max)

    #     # Frequency array calculation for each of the mass combination
    #     del_f = 1.0 / duration  # Extract scalar from array
    #     f_u = f_u + del_f
    #     fs = self.vmap_arange(f_l, f_u, del_f)

    #     hp, hc = self.gen_hphc(fs, theta_ripple, f_ref)

    #     return hp, hc, fs
        
    
    def noise_weighted_inner_product_jax(self, gw_param_dict, psd_list, detector_list, duration=None, duration_min=2, duration_max=128, npool=4, multiprocessing_verbose=True):
        """
        Compute the noise weighted inner product for a given waveform and PSD.

        Parameters
        ----------
        gw_param_dict: `dict`
            Dictionary containing the waveform parameters. The keys should be the parameter names and the values should be numpy arrays.
        psd_dict: bilby.gw.detector.PowerSpectralDensity object
            Dictionary containing the power spectral density for each detector.
        duration: `float` or `numpy.ndarray`
            Duration of the waveform. 
            Default is None. It will compute the duration based on the chirp time.
        duration_min: `float`
            Minimum duration of the waveform.
            Default is 2s.
        duration_max: `float`
            Maximum duration of the waveform.
            Default is 512s.
        verbose: `bool`
            If True, print the waveform parameters and PSDs.
            Default is False.

        Returns
        -------
        hp_inner_hp: `numpy.ndarray`
            Noise weighted inner product of h+ with h+
        """

        # set up GW parameters
        gw_param_dict = gw_param_dict.copy()
        size = len(gw_param_dict['mass_1'])
        # check dict values
        # for gw_param_dict, values should be an array
        for key, value in gw_param_dict.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Expected numpy array for {key} in gw_param_dict")
            else:
                gw_param_dict[key] = jnp.array(value)

        # set up waveform calculator inputs
        gw_param_dict['Mchirp'],  gw_param_dict['eta'] = ms_to_Mc_eta(jnp.array([ gw_param_dict['mass_1'], gw_param_dict['mass_2'] ]))
        gw_param_dict['tc'] = jnp.zeros(size)

        # set up input arguments for the JAX waveform generator 
        theta_ripple = []
        for key in self.arg_list:
            if key not in gw_param_dict.keys():
                raise ValueError(f"Missing key {key} in gw_param_dict")
            else:
                theta_ripple.append(gw_param_dict[key])
        theta_ripple = jnp.array(theta_ripple).T  # JAX numpy array

        # Set up duration of the signal
        if duration is not None:
            if isinstance(duration, float):
                duration = duration * np.ones(size)
            elif isinstance(duration, np.ndarray):
                if len(duration) != size:
                    raise ValueError(f"Duration array length should be equal to the number of mass combinations")
            else:
                raise ValueError(f"Duration should be either a float or a numpy array")
        else:
            # IMPORTANT: time duration calculation for each of the mass combination
            safety = 1.1
            mass_1 = jnp.array([gw_param_dict['mass_1']])
            mass_2 = jnp.array([gw_param_dict['mass_2']])
            f_l = float(self.f_l)
            approx_duration = safety * self.vmap_findchirp_chirptime(mass_1, mass_2, f_l)  # vmap result give you a stack which is 2D JAX array
            duration = np.ceil(approx_duration + 2.0).flatten()  # coverts to numpy 1D array
            
            if duration_max:
                duration[duration > duration_max] = duration_max  # IMRPheonomXPHM has maximum duration of 371s
            if duration_min:
                duration[duration < duration_min] = duration_min

        # Frequency array calculation for each of the mass combination
        del_f = 1.0 / duration
        fs = []
        # Frequency array size should be same for all events 
        # padding np.nan elements will be used to achieve this
        fsize_max = int((self.f_u - 0.0) / min(del_f)) + 2  # +2 to include upper limit
        fsize_arr = []
        for df  in del_f:
            # NOTE: frequency array start from 0Hz but h+,hx will be set to zero below fmin or self.f_l  
            flist = np.arange(0.0, self.f_u + df, df)
            fsize_arr.append(len(flist))  # record the array size to know how much padding will be added
            # add np.nan padding elements 
            if len(flist) <= fsize_max:
                # Add NaN values to make the array size equal to fsize_max
                flist = np.concatenate([flist, np.full(fsize_max - len(flist), np.nan)])
            fs.append(flist)

        # Convert list of arrays to a 2D JAX array
        fs = jnp.array(fs)

        # set up reference_frequency for vmap input
        f_ref = float(self.f_ref)
        # compute the waveform h+ and hx
        # NOTE: result will be in complex64 instead of complex128. This will be change later
        # vmap+jax.jit faster than just using jax.jit
        # mp.set_start_method('spawn', force=True)
        hp, hc = self.vmap_waveform(fs, theta_ripple, f_ref)
        hp, hc = np.array(hp, dtype=np.complex128), np.array(hc, dtype=np.complex128)
        # print(f"fs : {fs}")
        # print(f"theta_ripple : {theta_ripple}")
        # print(f"hp : {hp}")
        # print(f"hc : {hc}")
        fs = np.array(fs, dtype=np.float64)

        input_arguments = [
            (hp_i, hc_i, fs_i, fsize_arr_i, self.f_l, duration_i, idx, psd_list)
            for idx, (hp_i, hc_i, fs_i, fsize_arr_i, duration_i)
                in enumerate(zip(hp, hc, fs, fsize_arr, duration))
        ]

        num_det = len(detector_list)
        # np.shape(hp_inner_hp) = (len(num_det), size1)
        hp_inner_hp = np.zeros((num_det, size), dtype=np.complex128)
        hc_inner_hc = np.zeros((num_det, size), dtype=np.complex128)

        # for j in range(size):
        #     result = noise_weighted_inner_prod_ripple(input_arguments[j])
        #     hp_inner_hp_i, hc_inner_hc_i, iter_i = result
        #     hp_inner_hp[:, iter_i] = hp_inner_hp_i
        #     hc_inner_hc[:, iter_i] = hc_inner_hc_i

        # mp.set_start_method('fork', force=True)

        with Pool(processes=npool) as pool:
            # call the same function with different data in parallel
            # imap->retain order in the list, while map->doesn't
            if multiprocessing_verbose:
                for result in tqdm(
                    pool.imap_unordered(noise_weighted_inner_prod_ripple, input_arguments),
                    total=len(input_arguments),
                    ncols=100,
                ):
                    # but, np.shape(hp_inner_hp_i) = (size1, len(num_det))
                    hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                    hp_inner_hp[:, iter_i] = hp_inner_hp_i
                    hc_inner_hc[:, iter_i] = hc_inner_hc_i
            else:
                # with map, without tqdm
                for result in pool.map(noise_weighted_inner_prod_ripple, input_arguments):
                    hp_inner_hp_i, hc_inner_hc_i, iter_i = result
                    hp_inner_hp[:, iter_i] = hp_inner_hp_i
                    hc_inner_hc[:, iter_i] = hc_inner_hc_i
        # mp.set_start_method('spawn', force=True)

        return hp_inner_hp, hc_inner_hc

    # def noise_weighted_inner_product_jax(self, gw_param_dict, psd_list, detector_list, duration=None, duration_min=2, duration_max=128, npool=4, multiprocessing_verbose=True):
    #     """
    #     Compute the noise weighted inner product for a given waveform and PSD.

    #     Parameters
    #     ----------
    #     gw_param_dict: `dict`
    #         Dictionary containing the waveform parameters. The keys should be the parameter names and the values should be numpy arrays.
    #     psd_dict: bilby.gw.detector.PowerSpectralDensity object
    #         Dictionary containing the power spectral density for each detector.
    #     duration: `float` or `numpy.ndarray`
    #         Duration of the waveform. 
    #         Default is None. It will compute the duration based on the chirp time.
    #     duration_min: `float`
    #         Minimum duration of the waveform.
    #         Default is 2s.
    #     duration_max: `float`
    #         Maximum duration of the waveform.
    #         Default is 512s.
    #     verbose: `bool`
    #         If True, print the waveform parameters and PSDs.
    #         Default is False.

    #     Returns
    #     -------
    #     hp_inner_hp: `numpy.ndarray`
    #         Noise weighted inner product of h+ with h+
    #     """

    #     # set up GW parameters
    #     gw_param_dict = gw_param_dict.copy()
    #     size = len(gw_param_dict['mass_1'])
    #     # check dict values
    #     # for gw_param_dict, values should be an array
    #     for key, value in gw_param_dict.items():
    #         if not isinstance(value, np.ndarray):
    #             raise ValueError(f"Expected numpy array for {key} in gw_param_dict")
    #         else:
    #             gw_param_dict[key] = jnp.array(value)

    #     # set up waveform calculator inputs
    #     gw_param_dict['Mchirp'],  gw_param_dict['eta'] = ms_to_Mc_eta(jnp.array([ gw_param_dict['mass_1'], gw_param_dict['mass_2'] ]))
    #     gw_param_dict['tc'] = jnp.zeros(size)

    #     # set up input arguments for the JAX waveform generator 
    #     theta_ripple = []
    #     for key in self.arg_list:
    #         if key not in gw_param_dict.keys():
    #             raise ValueError(f"Missing key {key} in gw_param_dict")
    #         else:
    #             theta_ripple.append(gw_param_dict[key])
    #     theta_ripple = jnp.array(theta_ripple).T  # JAX numpy array

    #     # Set up duration of the signal
    #     if duration is not None:
    #         if isinstance(duration, float):
    #             duration = duration * np.ones(size)
    #         elif isinstance(duration, np.ndarray):
    #             if len(duration) != size:
    #                 raise ValueError(f"Duration array length should be equal to the number of mass combinations")
    #         else:
    #             raise ValueError(f"Duration should be either a float or a numpy array")
    #     else:
    #         # IMPORTANT: time duration calculation for each of the mass combination
    #         safety = 1.1
    #         mass_1 = jnp.array([gw_param_dict['mass_1']]).T
    #         mass_2 = jnp.array([gw_param_dict['mass_2']]).T
    #         f_l = self.f_l*jnp.ones((size,1))
    #         approx_duration = safety * self.vmap_findchirp_chirptime(mass_1, mass_2, f_l)  # vmap result give you a stack which is 2D JAX array
    #         duration = np.ceil(approx_duration + 2.0).flatten()  # coverts to numpy 1D array
            
    #         if duration_max:
    #             duration[duration > duration_max] = duration_max  # IMRPheonomXPHM has maximum duration of 371s
    #         if duration_min:
    #             duration[duration < duration_min] = duration_min

    #     # Frequency array calculation for each of the mass combination
    #     del_f = 1.0 / duration
    #     fs = []
    #     # Frequency array size should be same for all events 
    #     # padding np.nan elements will be used to achieve this
    #     fsize_max = int((self.f_u - 0.0) / min(del_f)) + 2  # +2 to include upper limit
    #     fsize_arr = []
    #     for df  in del_f:
    #         # NOTE: frequency array start from 0Hz but h+,hx will be set to zero below fmin or self.f_l  
    #         flist = np.arange(0.0, self.f_u + df, df)
    #         fsize_arr.append(len(flist))  # record the array size to know how much padding will be added
    #         # add np.nan padding elements 
    #         if len(flist) <= fsize_max:
    #             # Add NaN values to make the array size equal to fsize_max
    #             flist = np.concatenate([flist, np.full(fsize_max - len(flist), np.nan)])
    #         fs.append(flist)
    #     # Convert list of arrays to a 2D JAX array
    #     fs = jnp.array(fs)

    #     # set up reference_frequency for vmap input
    #     f_ref = self.f_ref * jnp.ones((size,1))
    #     # compute the waveform h+ and hx
    #     # NOTE: result will be in complex64 instead of complex128. This will be change later
    #     # vmap+jax.jit faster than just using jax.jit
    #     # mp.set_start_method('spawn', force=True)
    #     hp, hc = self.vmap_waveform(fs, theta_ripple, f_ref)
    #     hp, hc = np.array(hp, dtype=np.complex128), np.array(hc, dtype=np.complex128)
    #     fs = np.array(fs, dtype=np.float64)

    #     input_arguments = []
    #     for i in range(size):
    #         input_arguments.append(
    #             [
    #                 hp[i],
    #                 hc[i],
    #                 fs[i],
    #                 fsize_arr[i],
    #                 self.f_l,
    #                 duration[i],
    #                 i,
    #                 psd_list,
    #             ]
    #         )
    #     input_arguments = np.array(input_arguments, dtype=object)

    #     num_det = len(detector_list)
    #     # np.shape(hp_inner_hp) = (len(num_det), size1)
    #     hp_inner_hp = np.zeros((num_det, size), dtype=np.complex128)
    #     hc_inner_hc = np.zeros((num_det, size), dtype=np.complex128)

    #     # for j in range(size):
    #     #     result = noise_weighted_inner_prod_ripple(input_arguments[j])
    #     #     hp_inner_hp_i, hc_inner_hc_i, iter_i = result
    #     #     hp_inner_hp[:, iter_i] = hp_inner_hp_i
    #     #     hc_inner_hc[:, iter_i] = hc_inner_hc_i

    #     # mp.set_start_method('fork', force=True)

    #     with Pool(processes=npool) as pool:
    #         # call the same function with different data in parallel
    #         # imap->retain order in the list, while map->doesn't
    #         if multiprocessing_verbose:
    #             for result in tqdm(
    #                 pool.imap_unordered(noise_weighted_inner_prod_ripple, input_arguments),
    #                 total=len(input_arguments),
    #                 ncols=100,
    #             ):
    #                 # but, np.shape(hp_inner_hp_i) = (size1, len(num_det))
    #                 hp_inner_hp_i, hc_inner_hc_i, iter_i = result
    #                 hp_inner_hp[:, iter_i] = hp_inner_hp_i
    #                 hc_inner_hc[:, iter_i] = hc_inner_hc_i
    #         else:
    #             # with map, without tqdm
    #             for result in pool.map(noise_weighted_inner_prod_ripple, input_arguments):
    #                 hp_inner_hp_i, hc_inner_hc_i, iter_i = result
    #                 hp_inner_hp[:, iter_i] = hp_inner_hp_i
    #                 hc_inner_hc[:, iter_i] = hc_inner_hc_i
    #     # mp.set_start_method('spawn', force=True)

    #     return hp_inner_hp, hc_inner_hc
        

        # return input_arguments


        # # set up psd
        # psd_dict = psd_dict.copy()
        # for key, value in psd_dict.items(): # key is the detector name
        #     if not isinstance(value, bilby.gw.detector.PowerSpectralDensity):
        #         raise ValueError(f"Expected bilby.gw.detector.PowerSpectralDensity object for {key} in psd_dict")
        #     # compute the power spectral density array according to the frequency array
        #     psd_list = []
        #     fs_ = []
        #     for i in range(size):
        #         fs_.append(fs[i][:fsize_arr[i]])  # avoid np.nan paddings
        #         psd_list.append(psd_dict[key].get_power_spectral_density_array(fs_[i]))
        #     psd_dict[key] = np.array(psd_list, dtype=object)
        #     fs = np.array(fs_, dtype=object)
    
        # # set up h+,hx for inner_product
        # hp_, hc_ = [], []
        # for i in range(size):
        #     # remove the np.nan padding
        #     hp_.append(np.array(hp[i][:fsize_arr[i]], dtype=np.complex128))
        #     hc_.append(np.array(hc[i][:fsize_arr[i]], dtype=np.complex128))
        #     # find the index of 20Hz or nearby
        #     # set all elements to zero below this index
        #     idx = np.abs(fs[i] - self.f_l).argmin()
        #     hp_[i][0:idx] = 0.0 + 0.0j
        #     hc_[i][0:idx] = 0.0 + 0.0j
        # # each row don't have the same length, so keep the array type to object
        # hp = np.array(hp_, dtype=object)
        # hc = np.array(hc_, dtype=object)
        # #del hp_, hc_  # free up memory


        # # compute the noise weighted inner product
        # hp_inner_hp_list = []
        # hc_inner_hc_list = []
        # for det, psd_list in psd_dict.items():
        #     hp_inner_hp = []
        #     hc_inner_hc = []
        #     for i in range(size):  # loop over parameters
        #         psd_ = np.array(psd_list[i])
        #         duration_ = duration[i]
        #         idx2 = (psd_ != 0.0) & (psd_ != np.inf)  # this is necessary to avoid problem in np.sum 
        #         # it's cumbersome to use jax jitting noise_weighted_inner_product
        #         # so, I will use numba jitted one instead
        #         hp_inner_hp.append(self.vmap_noise_weighted_inner_product(
        #             hp[i][idx2],
        #             hp[i][idx2],
        #             psd_[idx2],
        #             duration_,
        #         ))

        #         hc_inner_hc.append(noise_weighted_inner_product(
        #             hc[i][idx2],
        #             hc[i][idx2],
        #             psd_[idx2],
        #             duration_,
        #         ))

        #     hp_inner_hp_list.append(hp_inner_hp)
        #     hc_inner_hc_list.append(hc_inner_hc)

        # hp_inner_hp = np.array(hp_inner_hp_list)
        # hc_inner_hc = np.array(hc_inner_hc_list)



        # # del psd_dict
        # # del gw_param_dict

        # return hp_inner_hp, hc_inner_hc