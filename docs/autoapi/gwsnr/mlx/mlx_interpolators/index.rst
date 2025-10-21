:py:mod:`gwsnr.mlx.mlx_interpolators`
=====================================

.. py:module:: gwsnr.mlx.mlx_interpolators


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.mlx.mlx_interpolators.find_index_1d_mlx
   gwsnr.mlx.mlx_interpolators.spline_interp_4pts_mlx
   gwsnr.mlx.mlx_interpolators.spline_interp_4x4x4x4pts_mlx
   gwsnr.mlx.mlx_interpolators.spline_interp_4x4x4x4pts_batched_mlx
   gwsnr.mlx.mlx_interpolators.get_interpolated_snr_aligned_spins_mlx
   gwsnr.mlx.mlx_interpolators.get_interpolated_snr_aligned_spins_helper
   gwsnr.mlx.mlx_interpolators.spline_interp_4x4pts_mlx
   gwsnr.mlx.mlx_interpolators.spline_interp_4x4pts_batched_mlx
   gwsnr.mlx.mlx_interpolators.get_interpolated_snr_no_spins_mlx
   gwsnr.mlx.mlx_interpolators.get_interpolated_snr_no_spins_helper



.. py:function:: find_index_1d_mlx(x_array, x_new)


.. py:function:: spline_interp_4pts_mlx(x_eval, x_pts, y_pts, condition_i)


.. py:function:: spline_interp_4x4x4x4pts_mlx(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new)

   
   Helper function that performs the FULL 4D interpolation for a SINGLE point.
   This function finds indices, slices data, and then uses vmap internally
   to perform interpolation efficiently without Python loops.
















   ..
       !! processed by numpydoc !!

.. py:function:: spline_interp_4x4x4x4pts_batched_mlx(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch)

   
   Perform batched 4D cubic spline interpolation using JAX vectorization.
















   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr_aligned_spins_mlx(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000)

   
   Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
   This function computes the SNR for gravitational wave signals with aligned spins across multiple
   detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
   and combines results from multiple detectors to produce the effective SNR.


   :Parameters:

       **mass_1** : jax.numpy.ndarray
           Primary mass of the binary system in solar masses.

       **mass_2** : jax.numpy.ndarray
           Secondary mass of the binary system in solar masses.

       **luminosity_distance** : jax.numpy.ndarray
           Luminosity distance to the source in Mpc.

       **theta_jn** : jax.numpy.ndarray
           Inclination angle between the orbital angular momentum and line of sight in radians.

       **psi** : jax.numpy.ndarray
           Polarization angle in radians.

       **geocent_time** : jax.numpy.ndarray
           GPS time of coalescence at the geocenter in seconds.

       **ra** : jax.numpy.ndarray
           Right ascension of the source in radians.

       **dec** : jax.numpy.ndarray
           Declination of the source in radians.

       **a_1** : jax.numpy.ndarray
           Dimensionless spin magnitude of the primary black hole.

       **a_2** : jax.numpy.ndarray
           Dimensionless spin magnitude of the secondary black hole.

       **detector_tensor** : jax.numpy.ndarray
           Detector tensor array containing detector response information.
           Shape: (n_detectors, ...)

       **snr_partialscaled** : jax.numpy.ndarray
           Pre-computed scaled partial SNR values for interpolation.
           Shape: (n_detectors, ...)

       **ratio_arr** : jax.numpy.ndarray
           Mass ratio grid points for interpolation (q = m2/m1).

       **mtot_arr** : jax.numpy.ndarray
           Total mass grid points for interpolation.

       **a1_arr** : jax.numpy.ndarray
           Primary spin grid points for interpolation.

       **a_2_arr** : jax.numpy.ndarray
           Secondary spin grid points for interpolation.

   :Returns:

       **snr** : jax.numpy.ndarray
           SNR values for each detector. Shape: (n_detectors, n_samples)

       **snr_effective** : jax.numpy.ndarray
           Effective SNR combining all detectors. Shape: (n_samples,)

       **snr_partial_** : jax.numpy.ndarray
           Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)

       **d_eff** : jax.numpy.ndarray
           Effective distance for each detector accounting for antenna response.
           Shape: (n_detectors, n_samples)








   .. rubric:: Notes

   - Uses 4D cubic spline interpolation for efficient SNR calculation
   - Assumes aligned spins (no precession)
   - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
   - Chirp mass and inclination-dependent factors are computed analytically





   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr_aligned_spins_helper(mass_1, mass_2, luminosity_distance, theta_jn, a_1, a_2, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, Fp, Fc, detector_tensor, batch_size)

   
   Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
   This function computes the SNR for gravitational wave signals with aligned spins across multiple
   detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
   and combines results from multiple detectors to produce the effective SNR.


   :Parameters:

       **mass_1** : jax.numpy.ndarray
           Primary mass of the binary system in solar masses.

       **mass_2** : jax.numpy.ndarray
           Secondary mass of the binary system in solar masses.

       **luminosity_distance** : jax.numpy.ndarray
           Luminosity distance to the source in Mpc.

       **theta_jn** : jax.numpy.ndarray
           Inclination angle between the orbital angular momentum and line of sight in radians.

       **psi** : jax.numpy.ndarray
           Polarization angle in radians.

       **geocent_time** : jax.numpy.ndarray
           GPS time of coalescence at the geocenter in seconds.

       **ra** : jax.numpy.ndarray
           Right ascension of the source in radians.

       **dec** : jax.numpy.ndarray
           Declination of the source in radians.

       **a_1** : jax.numpy.ndarray
           Dimensionless spin magnitude of the primary black hole.

       **a_2** : jax.numpy.ndarray
           Dimensionless spin magnitude of the secondary black hole.

       **detector_tensor** : jax.numpy.ndarray
           Detector tensor array containing detector response information.
           Shape: (n_detectors, ...)

       **snr_partialscaled** : jax.numpy.ndarray
           Pre-computed scaled partial SNR values for interpolation.
           Shape: (n_detectors, ...)

       **ratio_arr** : jax.numpy.ndarray
           Mass ratio grid points for interpolation (q = m2/m1).

       **mtot_arr** : jax.numpy.ndarray
           Total mass grid points for interpolation.

       **a1_arr** : jax.numpy.ndarray
           Primary spin grid points for interpolation.

       **a_2_arr** : jax.numpy.ndarray
           Secondary spin grid points for interpolation.

   :Returns:

       **snr** : jax.numpy.ndarray
           SNR values for each detector. Shape: (n_detectors, n_samples)

       **snr_effective** : jax.numpy.ndarray
           Effective SNR combining all detectors. Shape: (n_samples,)

       **snr_partial_** : jax.numpy.ndarray
           Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)

       **d_eff** : jax.numpy.ndarray
           Effective distance for each detector accounting for antenna response.
           Shape: (n_detectors, n_samples)








   .. rubric:: Notes

   - Uses 4D cubic spline interpolation for efficient SNR calculation
   - Assumes aligned spins (no precession)
   - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
   - Chirp mass and inclination-dependent factors are computed analytically





   ..
       !! processed by numpydoc !!

.. py:function:: spline_interp_4x4pts_mlx(q_array, mtot_array, snrpartialscaled_array, q_new, mtot_new)

   
   Helper function that performs the FULL 2D interpolation for a SINGLE point.
   This function finds indices, slices data, and then uses vmap internally
   to perform interpolation efficiently without Python loops.
















   ..
       !! processed by numpydoc !!

.. py:function:: spline_interp_4x4pts_batched_mlx(q_array, mtot_array, snrpartialscaled_array, q_new_batch, mtot_new_batch)

   
   Perform batched 2D cubic spline interpolation using MLX vectorization.
















   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr_no_spins_mlx(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr, batch_size=100000)

   
   Calculate interpolated signal-to-noise ratio (SNR) for aligned spin gravitational wave signals using JAX.
   This function computes the SNR for gravitational wave signals with aligned spins across multiple
   detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
   and combines results from multiple detectors to produce the effective SNR.


   :Parameters:

       **mass_1** : jax.numpy.ndarray
           Primary mass of the binary system in solar masses.

       **mass_2** : jax.numpy.ndarray
           Secondary mass of the binary system in solar masses.

       **luminosity_distance** : jax.numpy.ndarray
           Luminosity distance to the source in Mpc.

       **theta_jn** : jax.numpy.ndarray
           Inclination angle between the orbital angular momentum and line of sight in radians.

       **psi** : jax.numpy.ndarray
           Polarization angle in radians.

       **geocent_time** : jax.numpy.ndarray
           GPS time of coalescence at the geocenter in seconds.

       **ra** : jax.numpy.ndarray
           Right ascension of the source in radians.

       **dec** : jax.numpy.ndarray
           Declination of the source in radians.

       **a_1** : jax.numpy.ndarray
           Dimensionless spin magnitude of the primary black hole.

       **a_2** : jax.numpy.ndarray
           Dimensionless spin magnitude of the secondary black hole.

       **detector_tensor** : jax.numpy.ndarray
           Detector tensor array containing detector response information.
           Shape: (n_detectors, ...)

       **snr_partialscaled** : jax.numpy.ndarray
           Pre-computed scaled partial SNR values for interpolation.
           Shape: (n_detectors, ...)

       **ratio_arr** : jax.numpy.ndarray
           Mass ratio grid points for interpolation (q = m2/m1).

       **mtot_arr** : jax.numpy.ndarray
           Total mass grid points for interpolation.

       **a1_arr** : jax.numpy.ndarray
           Primary spin grid points for interpolation.

       **a_2_arr** : jax.numpy.ndarray
           Secondary spin grid points for interpolation.

   :Returns:

       **snr** : jax.numpy.ndarray
           SNR values for each detector. Shape: (n_detectors, n_samples)

       **snr_effective** : jax.numpy.ndarray
           Effective SNR combining all detectors. Shape: (n_samples,)

       **snr_partial_** : jax.numpy.ndarray
           Interpolated partial SNR values for each detector. Shape: (n_detectors, n_samples)

       **d_eff** : jax.numpy.ndarray
           Effective distance for each detector accounting for antenna response.
           Shape: (n_detectors, n_samples)








   .. rubric:: Notes

   - Uses 4D cubic spline interpolation for efficient SNR calculation
   - Assumes aligned spins (no precession)
   - Effective SNR is calculated as sqrt(sum(SNR_i^2)) across detectors
   - Chirp mass and inclination-dependent factors are computed analytically





   ..
       !! processed by numpydoc !!

.. py:function:: get_interpolated_snr_no_spins_helper(mass_1, mass_2, luminosity_distance, theta_jn, snr_partialscaled, ratio_arr, mtot_arr, Fp, Fc, detector_tensor, batch_size)

   
   Calculate interpolated signal-to-noise ratio (SNR) for non-spinning gravitational wave signals using JAX.
   This function computes the SNR for gravitational wave signals without spins across multiple
   detectors using 4D cubic spline interpolation. It calculates the effective distance, partial SNR,
   and combines results from multiple detectors to produce the effective SNR.
















   ..
       !! processed by numpydoc !!

