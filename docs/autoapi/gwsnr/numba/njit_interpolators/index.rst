:orphan:

:py:mod:`gwsnr.numba.njit_interpolators`
========================================

.. py:module:: gwsnr.numba.njit_interpolators


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.numba.njit_interpolators.find_index_1d_numba
   gwsnr.numba.njit_interpolators.cubic_function_4pts_numba
   gwsnr.numba.njit_interpolators.cubic_spline_4d_numba
   gwsnr.numba.njit_interpolators.cubic_spline_4d_batched_numba



.. py:function:: find_index_1d_numba(x_array, x_new)

   
   Find the index for cubic spline interpolation in 1D.
   Returns the index and a condition for edge handling.


   :Parameters:

       **x_array** : jnp.ndarray
           The array of x values for interpolation. Must be sorted in ascending order.

       **x_new** : float or jnp.ndarray
           The new x value(s) to find the index for.

   :Returns:

       **i** : int
           The index in `x_array` such that `x_array[i-1] <= x_new < x_array[i+1]`.

       **condition_i** : int
           An integer indicating the condition for edge handling:
           - 1: `x_new` is less than or equal to the first element of `x_array`.
           - 2: `x_new` is between the first and last elements of `x_array`.
           - 3: `x_new` is greater than or equal to the last element of `x_array`.








   .. rubric:: Notes

   Uses binary search with clipped indices to ensure valid 4-point stencils.
   The condition parameter determines linear vs cubic interpolation at boundaries.





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_function_4pts_numba(x_eval, x_pts, y_pts, condition_i)

   
   Evaluate a cubic function at a given point using 4-point interpolation.


   :Parameters:

       **x_eval** : float
           The x value at which to evaluate the cubic function.

       **x_pts** : jnp.ndarray
           The x values of the 4 points used for interpolation. Must be sorted in ascending order

       **y_pts** : jnp.ndarray
           The y values corresponding to the x_pts. Must have the same length as x_pts.

       **condition_i** : int
           An integer indicating the condition for edge handling:
           - 1: `x_eval` is less than or equal to the first element of `x_pts`.
           - 2: `x_eval` is between the first and last elements of `x_pts`.
           - 3: `x_eval` is greater than or equal to the last element of `x_pts`.

   :Returns:

       float
           The interpolated value at `x_eval`.








   .. rubric:: Notes

   This function uses cubic Hermite interpolation for the main case (condition_i == 2).
   For edge cases (condition_i == 1 or 3), it uses linear interpolation between the first two or last two points, respectively.
   The x_pts and y_pts must be of length 4, and x_pts must be sorted in ascending order.
   The function assumes that the input arrays are valid and does not perform additional checks.
   If the input arrays are not of length 4, or if x_pts is not sorted, the behavior is undefined.
   The function is designed to be used with Numba's JIT compilation for performance.
   It is optimized for speed and does not include error handling or input validation.
   The cubic Hermite interpolation is based on the tangents calculated from the y values at the four points.
   The tangents are computed using the differences between the y values and the x values
   to ensure smoothness and continuity of the interpolated curve.





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_4d_numba(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new, int_q, int_m, int_a1, int_a2)

   
   Perform cubic spline interpolation in 4D for the given arrays and new values.


   :Parameters:

       **q_array** : jnp.ndarray
           The array of q values for interpolation. Must be sorted in ascending order.

       **mtot_array** : jnp.ndarray
           The array of mtot values for interpolation. Must be sorted in ascending order.

       **a1_array** : jnp.ndarray
           The array of a1 values for interpolation. Must be sorted in ascending order.

       **a2_array** : jnp.ndarray
           The array of a2 values for interpolation. Must be sorted in ascending order.

       **snrpartialscaled_array** : jnp.ndarray
           The 4D array of snrpartialscaled values with shape (4, 4, 4, 4).
           This array contains the values to be interpolated.

       **q_new** : float
           The new q value at which to evaluate the cubic spline.

       **mtot_new** : float
           The new mtot value at which to evaluate the cubic spline.

       **a1_new** : float
           The new a1 value at which to evaluate the cubic spline.

       **a2_new** : float
           The new a2 value at which to evaluate the cubic spline.

       **int_q** : int
           edge condition for q interpolation. Refer to `find_index_1d_numba` for details.

       **int_m** : int
           edge condition for mtot interpolation. Refer to `find_index_1d_numba` for details.

       **int_a1** : int
           edge condition for a1 interpolation. Refer to `find_index_1d_numba` for details.

       **int_a2** : int
           edge condition for a2 interpolation. Refer to `find_index_1d_numba` for details.

   :Returns:

       float
           The interpolated value at the new coordinates (q_new, mtot_new, a1_new, a2_new).








   .. rubric:: Notes

   This function uses cubic Hermite interpolation for the main case (int_q == 2, int_m == 2, int_a1 == 2, int_a2 == 2).
   For edge cases (int_q == 1 or 3, int_m == 1 or 3, int_a1 == 1 or 3, int_a2 == 1 or 3), it uses linear interpolation between the first two or last two points, respectively.





   ..
       !! processed by numpydoc !!

.. py:function:: cubic_spline_4d_batched_numba(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch)

   
   Perform cubic spline interpolation in 4D for a batch of new values.


   :Parameters:

       **q_array** : jnp.ndarray
           The array of q values for interpolation. Must be sorted in ascending order.

       **mtot_array** : jnp.ndarray
           The array of mtot values for interpolation. Must be sorted in ascending order.

       **a1_array** : jnp.ndarray
           The array of a1 values for interpolation. Must be sorted in ascending order.

       **a2_array** : jnp.ndarray
           The array of a2 values for interpolation. Must be sorted in ascending order.

       **snrpartialscaled_array** : jnp.ndarray
           The 4D array of snrpartialscaled values.

       **q_new_batch** : jnp.ndarray
           The new q values at which to evaluate the cubic spline. Must be a 1D array.

       **mtot_new_batch** : jnp.ndarray
           The new mtot values at which to evaluate the cubic spline. Must be a 1
           The new a1 values at which to evaluate the cubic spline. Must be a 1D array.

       **a1_new_batch** : jnp.ndarray
           The new a1 values at which to evaluate the cubic spline. Must be a 1D array.

       **a2_new_batch** : jnp.ndarray
           The new a2 values at which to evaluate the cubic spline. Must be a 1D array.

   :Returns:

       jnp.ndarray
           A 1D array of interpolated values at the new coordinates (q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch).













   ..
       !! processed by numpydoc !!

