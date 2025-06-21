import numpy as np
import jax.numpy as jnp
from ripple import ms_to_Mc_eta
from jax import vmap
from jax import jit
from ..jax import findchirp_chirptime_jax
from multiprocessing import Pool
from tqdm import tqdm

from ..utils import noise_weighted_inner_prod_ripple

class RippleInnerProduct:
    """
    Class to compute the noise weighted inner product for a given waveform and PSD
    """

    def __init__(self, waveform_name, minimum_frequency=20.0, sampling_frequency=2048.0, reference_frequency=None):
        
        self.f_u = sampling_frequency/2
        self.f_ref = reference_frequency if reference_frequency is not None else minimum_frequency
        self.f_l = minimum_frequency
        self.arg_list = self.arg_selection(waveform_name)
        self.gen_hphc = jit(self.select_waveform(waveform_name))
        self.vmap_waveform = vmap(self.gen_hphc)
        self.vmap_findchirp_chirptime = vmap(findchirp_chirptime_jax)

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

    # JAX-compatible noise weighted inner product
    @staticmethod
    @jit
    def noise_weighted_inner_product(signal1, signal2, psd, duration):
        prod = jnp.conj(signal1) * signal2 / psd
        return 4.0 / duration * jnp.sum(prod)
    
    # JAX-compatible per-detector SNR inner product, vectorized over detectors
    @staticmethod
    @jit
    def snr_for_detector(hp, hc, psd_arr, duration):
        # All inputs are arrays for a given freq grid
        idx2 = (psd_arr != 0.0) & (psd_arr != jnp.inf)
        hp_inner = RippleInnerProduct.noise_weighted_inner_product(
            hp[idx2], hp[idx2], psd_arr[idx2], duration
        )
        hc_inner = RippleInnerProduct.noise_weighted_inner_product(
            hc[idx2], hc[idx2], psd_arr[idx2], duration
        )
        return hp_inner, hc_inner
    
    def noise_weighted_inner_product_jax(
        self, gw_param_dict, psd_array_list, duration=None, duration_min=2, duration_max=128
    ):
        """
        Compute the noise weighted inner product for all GW params and detectors, **fully JAX**.
        Inputs:
            gw_param_dict: dict of {key: jnp.ndarray}
            psd_array_list: list of jnp.ndarray, one for each detector (each shape: (freqs,))
            duration: float or jnp.ndarray of shape (N,)
        Returns:
            hp_inner_hp: shape (num_detectors, N)
            hc_inner_hc: shape (num_detectors, N)
        """
        gw_param_dict = {k: jnp.array(v) for k, v in gw_param_dict.items()}
        size = len(gw_param_dict['mass_1'])
        gw_param_dict['Mchirp'], gw_param_dict['eta'] = ms_to_Mc_eta(
            jnp.stack([gw_param_dict['mass_1'], gw_param_dict['mass_2']])
        )
        gw_param_dict['tc'] = jnp.zeros(size)

        # Construct the parameter matrix for the waveform generator
        theta_ripple = jnp.stack([gw_param_dict[k] for k in self.arg_list], axis=-1)
        # Duration calculation (as before)
        if duration is None:
            safety = 1.2
            m1 = gw_param_dict['mass_1'][:, None]
            m2 = gw_param_dict['mass_2'][:, None]
            f_l = self.f_l * jnp.ones((size, 1))
            approx_duration = safety * self.vmap_findchirp_chirptime(m1, m2, f_l).flatten()
            duration = jnp.clip(jnp.ceil(approx_duration + 2.0), duration_min, duration_max)
        else:
            duration = duration * jnp.ones(size) if isinstance(duration, float) else jnp.array(duration)

        # Frequency arrays and zero-padding
        del_f = 1.0 / duration
        fsize_max = int((self.f_u - 0.0) / jnp.min(del_f)) + 2
        def make_freqs(df):
            flist = jnp.arange(0.0, self.f_u + df, df)
            pad_size = fsize_max - flist.shape[0]
            flist = jnp.pad(flist, (0, pad_size), constant_values=jnp.nan)
            return flist
        fs = vmap(make_freqs)(del_f)
        f_ref = self.f_ref * jnp.ones((size, 1))

        # Generate waveforms
        hp, hc = self.vmap_waveform(fs, theta_ripple, f_ref)

        # For each detector, interpolate the PSD to match fs (or assume already matches)
        # psd_array_list: list of shape (num_det, fsize_max)
        psd_array = jnp.stack(psd_array_list)  # shape (num_det, fsize_max)

        # Now, we want to compute inner product for each event/detector
        # So vmap over (event) and (detector)
        # All shapes: (num_det, N, fsize_max)

        hp = hp.astype(jnp.complex64)
        hc = hc.astype(jnp.complex64)
        fs = fs.astype(jnp.float32)
        duration = duration.astype(jnp.float32)

        # Helper: slice/pad per-event arrays to common shape
        # Assume hp, hc, fs all have shape (N, fsize_max)
        def event_detector_inner(hp_row, hc_row, psd_row, duration_row):
            # Mask out nan freq bins
            mask = ~jnp.isnan(hp_row)
            hp_valid = jnp.where(mask, hp_row, 0.0 + 0.0j)
            hc_valid = jnp.where(mask, hc_row, 0.0 + 0.0j)
            psd_valid = jnp.where(mask, psd_row, 1e50)  # large noise for nans
            return RippleInnerProduct.snr_for_detector(hp_valid, hc_valid, psd_valid, duration_row)
        # vmap over detectors (outer), then over events (inner)
        batched_inner = vmap(
            vmap(event_detector_inner, in_axes=(0, 0, 0, 0)),  # over events
            in_axes=(None, None, 0, None)  # over detectors
        )
        # Arrange arguments for vmap: shape (N, fsize_max), (N, fsize_max), (num_det, fsize_max), (N,)
        # We need to broadcast psd_array to (num_det, N, fsize_max)
        hp_b = jnp.broadcast_to(hp, (len(psd_array_list), size, fsize_max))
        hc_b = jnp.broadcast_to(hc, (len(psd_array_list), size, fsize_max))
        duration_b = jnp.broadcast_to(duration, (len(psd_array_list), size))
        results = vmap(
            lambda psd: vmap(event_detector_inner, in_axes=(0, 0, None, 0))(hp, hc, psd, duration)
        )(psd_array)
        # results: (num_det, N, 2), where last dim is (hp_inner, hc_inner)
        hp_inner_hp, hc_inner_hc = results[..., 0], results[..., 1]
        return hp_inner_hp, hc_inner_hc