import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit, vmap, lax
from .jaxjit_functions import antenna_response_array

@jit
def find_index_1d_jax(x_array, x_new):
    N = x_array.shape[0]
    i = jnp.sum(x_array <= x_new) - 1
    i = jnp.clip(i, 1, N - 3)
    low_lim = x_array[0]+ (x_array[1]-x_array[0]) / 2.0
    high_lim = x_array[N-2] + (x_array[N-1] - x_array[N-2]) / 2.0
    
    condition_0 = x_new <= low_lim
    condition_1 = (x_new > low_lim) & (x_new <= x_array[1])
    condition_2 = (x_new > x_array[1]) & (x_new < x_array[N - 2])
    condition_3 = (x_new >= x_array[N - 2]) & (x_new < high_lim)
    condition_4 = x_new >= high_lim

    condition_i = jnp.where(
        condition_0, 0,
        jnp.where(condition_1, 1,
        jnp.where(condition_2, 2,
        jnp.where(condition_3, 3, 4))))
        
    return i, condition_i

@jit
def cubic_function_4pts_jax(x_eval, x_pts, y_pts, i):
    """
    A corrected and simplified JIT-compatible cubic spline function.

    This version uses a clear mapping from the input index 'i' to the
    appropriate branch for lax.switch.
    """
    # --- 1. Define a function for each logical branch ---
    # Each function takes the same 'operands' tuple as input. This is a
    # standard and required pattern for lax.switch.

    # Branch for i = 0
    def branch_0(operands):
        _, _, y_operands = operands
        return y_operands[0]

    # Branch for i = 1
    def branch_1(operands):
        _, _, y_operands = operands
        return y_operands[1]

    # The main "else" case for spline calculation (i = 2 or other values)
    def branch_else(operands):
        x_eval_op, x_op, y_op = operands
        
        matrixA = jnp.array([
            [x_op[0]**3, x_op[0]**2, x_op[0], 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
            [x_op[1]**3, x_op[1]**2, x_op[1], 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, x_op[1]**3, x_op[1]**2, x_op[1], 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, x_op[2]**3, x_op[2]**2, x_op[2], 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, x_op[2]**3, x_op[2]**2, x_op[2], 1.0],
            [0, 0, 0, 0, 0, 0, 0, 0, x_op[3]**3, x_op[3]**2, x_op[3], 1.0],
            [3*x_op[1]**2, 2*x_op[1], 1.0, 0, -3*x_op[1]**2, -2*x_op[1], -1.0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3*x_op[2]**2, 2*x_op[2], 1.0, 0, -3*x_op[2]**2, -2*x_op[2], -1.0, 0],
            [6*x_op[1], 2.0, 0.0, 0.0, -6*x_op[1], -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 6*x_op[2], 2.0, 0.0, 0.0, -6*x_op[2], -2.0, 0.0, 0.0],
            [6*x_op[0], 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 6*x_op[3], 2.0, 0, 0],
        ])
        matrixC = jnp.array([
            y_op[0], y_op[1], y_op[1], y_op[2], y_op[2], y_op[3],
            0, 0, 0, 0, 0, 0
        ])
        coeffs = jnp.linalg.solve(matrixA, matrixC)
        snr = coeffs[4]*x_eval_op**3 + coeffs[5]*x_eval_op**2 + coeffs[6]*x_eval_op + coeffs[7]
        return snr

    # Branch for i = 3
    def branch_3(operands):
        _, _, y_operands = operands
        return y_operands[2]

    # Branch for i = 4
    def branch_4(operands):
        _, _, y_operands = operands
        return y_operands[3]

    # --- 2. Map input 'i' to a branch index (0-4) ---
    # This uses jnp.where to create a JIT-compatible mapping. If i is not
    # one of 0, 1, 3, or 4, it defaults to 2, which calls our 'branch_else'.
    # This is a more readable version of your lax.select chain.
    branch_index = jnp.where(i == 0, 0,
                   jnp.where(i == 1, 1,
                   jnp.where(i == 3, 3,
                   jnp.where(i == 4, 4, 2)))) # Default index is 2

    # --- 3. Call lax.switch with the ordered branches and data ---
    ordered_branches = [branch_0, branch_1, branch_else, branch_3, branch_4]
    
    # Bundle all the data needed by any branch into a single tuple.
    operands = (x_eval, x_pts, y_pts)
    
    return lax.switch(branch_index, ordered_branches, operands)

######### Interpolation 4D #########
@jit
def natural_cubic_spline_4d_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
    q_idx, int_q = find_index_1d_jax(q_array, q_new)
    m_idx, int_m = find_index_1d_jax(mtot_array, mtot_new)
    a_1, int_a1 = find_index_1d_jax(a1_array, a1_new)
    a_2, int_a2 = find_index_1d_jax(a2_array, a2_new)

    F = lax.dynamic_slice(snrpartialscaled_array, (q_idx - 1, m_idx - 1, a_1 - 1, a_2 - 1), (4, 4, 4, 4))
    qs = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
    ms = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
    a1s = lax.dynamic_slice(a1_array, (a_1 - 1,), (4,))
    a2s = lax.dynamic_slice(a2_array, (a_2 - 1,), (4,))

    partialsnr_q = jnp.zeros(4)
    for i in range(4):
        partialsnr_m = jnp.zeros(4)
        for j in range(4):
            partialsnr_a1 = jnp.zeros(4)
            for k in range(4):
                # Interpolate along a2 (last axis)
                partialsnr_a1= partialsnr_a1.at[k].set(cubic_function_4pts_jax(a2_new, a2s, F[i, j, k, :], int_a2))
            # Interpolate along a1
            partialsnr_m = partialsnr_m.at[j].set(cubic_function_4pts_jax(a1_new, a1s, partialsnr_a1, int_a1))
        # Interpolate along mtot
        partialsnr_q = partialsnr_q.at[i].set(cubic_function_4pts_jax(mtot_new, ms, partialsnr_m, int_m))
    # Interpolate along q
    snr_new = cubic_function_4pts_jax(q_new, qs, partialsnr_q, int_q)
    return snr_new

# Vectorized version
@jit
def natural_cubic_spline_4d_batched_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_batch, mtot_batch, a1_batch, a2_batch):
    # vmapped_interp = vmap(lambda q, m: spline_4d_interp_single(q_array, mtot_array, snrpartialscaled_array, q, m), in_axes=(0, 0))
    vmapped_interp = vmap(lambda q, m, a1, a2: natural_cubic_spline_4d_jax(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q, m, a1, a2), in_axes=(0, 0, 0, 0))
    return vmapped_interp(q_batch, mtot_batch, a1_batch, a2_batch)

@jit
def get_interpolated_snr_aligned_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr):
    """
    Function to calculate the interpolated snr for a given set of parameters
    """

    size = mass_1.shape[0]
    len_ = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    # get array of antenna response
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = jnp.cos(theta_jn) ** 2
    ci_param = ((1 + jnp.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = jnp.zeros((len_,size))
    d_eff = jnp.zeros((len_,size))
    snr = jnp.zeros((len_,size))
    # loop over the detectors
    for j in range(len_):
        snr_partial_ = snr_partial_.at[j].set(
            jnp.array(natural_cubic_spline_4d_batched_jax(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                a1_array=a1_arr,
                a2_array=a_2_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_batch=ratio,
                mtot_batch=mtot,
                a1_batch=a_1,
                a2_batch= a_2,
            ))
        )

        # calculate the effective distance
        # d_eff[j] =luminosity_distance / jnp.sqrt(
        #             Fp[j]**2 * ci_param + Fc[j]**2 * ci_2
        #         )
        d_eff = d_eff.at[j].set(
            luminosity_distance / jnp.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        )
        # snr[j] = snr_partial_buffer * A1 / d_eff[j]
        snr = snr.at[j].set(
            snr_partial_[j] * A1 / d_eff[j]
        )
    
    snr_effective = jnp.sqrt(jnp.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff

######### Interpolation 2D #########
@jit
def natural_cubic_spline_2d_jax(q_array, mtot_array, snrpartialscaled_array, q_new, mtot_new):
    q_idx, int_q = find_index_1d_jax(q_array, q_new)
    m_idx, int_m = find_index_1d_jax(mtot_array, mtot_new)

    F = lax.dynamic_slice(snrpartialscaled_array, (q_idx - 1, m_idx - 1), (4, 4))
    qs = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
    ms = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))

    partialsnr_q = jnp.zeros(4)
    for i in range(4):
        partialsnr_m = jnp.zeros(4)
        # Interpolate along mtot
        partialsnr_q = partialsnr_q.at[i].set(cubic_function_4pts_jax(mtot_new, ms, F[i, :], int_m))
    # Interpolate along q
    snr_new = cubic_function_4pts_jax(q_new, qs, partialsnr_q, int_q)
    return snr_new

# Vectorized version
@jit
def natural_cubic_spline_2d_batched_jax(q_array, mtot_array, snrpartialscaled_array, q_batch, mtot_batch):
    vmapped_interp = vmap(lambda q, m: natural_cubic_spline_2d_jax(q_array, mtot_array, snrpartialscaled_array, q, m), in_axes=(0, 0))
    return vmapped_interp(q_batch, mtot_batch)

@jit
def get_interpolated_snr_no_spins_jax(mass_1, mass_2, luminosity_distance, theta_jn, psi, geocent_time, ra, dec, a_1, a_2, detector_tensor, snr_partialscaled, ratio_arr, mtot_arr, a1_arr, a_2_arr):
    """
    Function to calculate the in terpolated snr for a given set of parameters
    """

    size = mass_1.shape[0]
    len_ = detector_tensor.shape[0]
    mtot = mass_1 + mass_2
    ratio = mass_2 / mass_1

    # get array of antenna response
    Fp, Fc = antenna_response_array(ra, dec, geocent_time, psi, detector_tensor)

    Mc = ((mass_1 * mass_2) ** (3 / 5)) / ((mass_1 + mass_2) ** (1 / 5))
    A1 = Mc ** (5.0 / 6.0)
    ci_2 = jnp.cos(theta_jn) ** 2
    ci_param = ((1 + jnp.cos(theta_jn) ** 2) / 2) ** 2

    snr_partial_ = jnp.zeros((len_,size))
    d_eff = jnp.zeros((len_,size))
    snr = jnp.zeros((len_,size))
    # loop over the detectors
    for j in range(len_):
        snr_partial_ = snr_partial_.at[j].set(
            jnp.array(natural_cubic_spline_2d_batched_jax(
                q_array=ratio_arr,
                mtot_array=mtot_arr,
                snrpartialscaled_array=snr_partialscaled[j],
                q_batch=ratio,
                mtot_batch=mtot,
            ))
        )

        d_eff = d_eff.at[j].set(
            luminosity_distance / jnp.sqrt(Fp[j] ** 2 * ci_param + Fc[j] ** 2 * ci_2)
        )
        # snr[j] = snr_partial_buffer * A1 / d_eff[j]
        snr = snr.at[j].set(
            snr_partial_[j] * A1 / d_eff[j]
        )
    
    snr_effective = jnp.sqrt(jnp.sum(snr ** 2, axis=0))

    return snr, snr_effective, snr_partial_, d_eff

# ########### For testing only ##################
# def catmull_rom_spline(p, t):
    
#     M = 0.5 * jnp.array([
#         [0,  2,  0,  0],
#         [-1, 0,  1,  0],
#         [2, -5,  4, -1],
#         [-1, 3, -3,  1]
#     ])
#     T = jnp.array([1.0, t, t**2, t**3])
#     return T @ M @ p

# @jit
# def find_index_1d(x_array, x_new):
#     N = x_array.shape[0]
#     i = jnp.sum(x_array <= x_new) - 1
#     return jnp.clip(i, 1, N - 3)

# @jit
# def catmull_rom_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new, mtot_new, a1_new, a2_new):
#     q_idx = find_index_1d(q_array, q_new)
#     m_idx = find_index_1d(mtot_array, mtot_new)
#     a1_idx = find_index_1d(a1_array, a1_new)
#     a2_idx = find_index_1d(a2_array, a2_new)

#     # Dynamic slices (for 4x4 neighborhood)
#     F = lax.dynamic_slice(snrpartialscaled_array, (q_idx-1, m_idx-1, a1_idx-1, a2_idx-1), (4,4,4,4))
#     qs = lax.dynamic_slice(q_array, (q_idx - 1,), (4,))
#     ms = lax.dynamic_slice(mtot_array, (m_idx - 1,), (4,))
#     a1s = lax.dynamic_slice(a1_array, (a1_idx - 1,), (4,))
#     a2s = lax.dynamic_slice(a2_array, (a2_idx - 1,), (4,))

#     # Relative coordinates
#     tq = (q_new - qs[1]) / (qs[2] - qs[1])
#     tm = (mtot_new - ms[1]) / (ms[2] - ms[1])
#     ta1 = (a1_new - a1s[1]) / (a1s[2] - a1s[1])
#     ta2 = (a2_new - a2s[1]) / (a2s[2] - a2s[1])

#     # Tricubic interpolation logic, extended to 4D:
#     temp_q = jnp.zeros(4)
#     for i in range(4):
#         temp_m = jnp.zeros(4)
#         for j in range(4):
#             temp_a1 = jnp.zeros(4)
#             for k in range(4):
#                 # Interpolate along a2 (last axis)
#                 temp_a1= temp_a1.at[k].set(catmull_rom_spline(F[i, j, k, :], ta2))
#             # Interpolate along a1
#             temp_m = temp_m.at[j].set(catmull_rom_spline(temp_a1, ta1))
#         # Interpolate along mtot
#         temp_q = temp_q.at[i].set(catmull_rom_spline(temp_m, tm))
#     # Interpolate along q
#     snr_new = catmull_rom_spline(temp_q, tq)
#     return snr_new

# @jit
# def batched_catmull_rom_4d(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch):
#     # Vectorize only over q_new and mtot_new
#     vmapped_interp = vmap(
#         lambda q, m, a1, a2: catmull_rom_4d_interp_single(q_array, mtot_array, a1_array, a2_array, snrpartialscaled_array, q, m, a1, a2),
#         in_axes=(0, 0, 0, 0)
#     )
#     return vmapped_interp(q_new_batch, mtot_new_batch, a1_new_batch, a2_new_batch)


# #################