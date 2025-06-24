"""
Optimized 4D interpolation with pre-computed coefficients
"""
import numpy as np
from numba import njit

@njit
def precompute_cubic_coefficients_1d(x_array, y_array):
    """
    Pre-compute cubic spline coefficients for 1D interpolation.
    Returns coefficients [a, b, c, d] for each interval.
    """
    n = len(x_array)
    h = np.diff(x_array)
    
    # Build tridiagonal system for second derivatives
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Natural spline boundary conditions
    A[0, 0] = 1.0
    A[n-1, n-1] = 1.0
    
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2.0 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3.0 * ((y_array[i+1] - y_array[i]) / h[i] - 
                      (y_array[i] - y_array[i-1]) / h[i-1])
    
    # Solve for second derivatives
    c = np.linalg.solve(A, b)
    
    # Compute coefficients for each interval
    coeffs = np.zeros((n-1, 4))  # [a, b, c, d] for each interval
    
    for i in range(n-1):
        dx = h[i]
        coeffs[i, 0] = y_array[i]  # d (constant term)
        coeffs[i, 1] = (y_array[i+1] - y_array[i]) / dx - dx * (2*c[i] + c[i+1]) / 3.0  # c (linear)
        coeffs[i, 2] = c[i]  # b (quadratic)
        coeffs[i, 3] = (c[i+1] - c[i]) / (3.0 * dx)  # a (cubic)
    
    return coeffs

@njit
def fast_cubic_eval(x_eval, x_array, coeffs):
    """
    Fast cubic spline evaluation using pre-computed coefficients.
    """
    # Find interval
    i = np.searchsorted(x_array, x_eval, side='right') - 1
    i = max(0, min(len(coeffs)-1, i))
    
    # Evaluate polynomial
    dx = x_eval - x_array[i]
    return coeffs[i, 0] + coeffs[i, 1]*dx + coeffs[i, 2]*dx*dx + coeffs[i, 3]*dx*dx*dx

@njit
def optimized_4d_interpolation(q_array, mtot_array, a1_array, a2_array, 
                              data_4d, q_new, mtot_new, a1_new, a2_new):
    """
    Optimized 4D interpolation using tensor product approach.
    """
    # Find grid indices
    q_idx = np.searchsorted(q_array, q_new, side='right') - 1
    m_idx = np.searchsorted(mtot_array, mtot_new, side='right') - 1  
    a1_idx = np.searchsorted(a1_array, a1_new, side='right') - 1
    a2_idx = np.searchsorted(a2_array, a2_new, side='right') - 1
    
    # Clamp indices
    q_idx = max(1, min(len(q_array)-3, q_idx))
    m_idx = max(1, min(len(mtot_array)-3, m_idx))
    a1_idx = max(1, min(len(a1_array)-3, a1_idx))
    a2_idx = max(1, min(len(a2_array)-3, a2_idx))
    
    # Extract 4x4x4x4 local grid
    q_local = q_array[q_idx-1:q_idx+3]
    m_local = mtot_array[m_idx-1:m_idx+3]
    a1_local = a1_array[a1_idx-1:a1_idx+3]
    a2_local = a2_array[a2_idx-1:a2_idx+3]
    data_local = data_4d[q_idx-1:q_idx+3, m_idx-1:m_idx+3, 
                        a1_idx-1:a1_idx+3, a2_idx-1:a2_idx+3]
    
    # Pre-compute coefficients for each 1D slice
    # This is still expensive but reduces repeated computation
    result = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Linear interpolation along a2 (fastest varying)
                w_a2 = lagrange_weight_4pt(a2_new, a2_local)
                val_a2 = np.sum(w_a2 * data_local[i, j, k, :])
                
                # Accumulate with weights for other dimensions
                w_a1 = lagrange_weight_4pt(a1_new, a1_local)[k]
                w_m = lagrange_weight_4pt(mtot_new, m_local)[j] 
                w_q = lagrange_weight_4pt(q_new, q_local)[i]
                
                result += w_q * w_m * w_a1 * val_a2
                
    return result

@njit 
def lagrange_weight_4pt(x, x_pts):
    """
    Compute Lagrange interpolation weights for 4 points.
    More stable than solving linear systems.
    """
    weights = np.zeros(4)
    for i in range(4):
        w = 1.0
        for j in range(4):
            if i != j:
                w *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        weights[i] = w
    return weights

# Alternative: Tensor product Lagrange interpolation
@njit
def tensor_product_4d_lagrange(q_array, mtot_array, a1_array, a2_array,
                              data_4d, q_new, mtot_new, a1_new, a2_new):
    """
    Fastest approach: Direct tensor product using Lagrange polynomials.
    """
    # Find indices and extract local 4-point grids
    q_idx = np.searchsorted(q_array, q_new) - 1
    m_idx = np.searchsorted(mtot_array, mtot_new) - 1
    a1_idx = np.searchsorted(a1_array, a1_new) - 1  
    a2_idx = np.searchsorted(a2_array, a2_new) - 1
    
    # Clamp to valid range
    q_idx = max(1, min(len(q_array)-3, q_idx))
    m_idx = max(1, min(len(mtot_array)-3, m_idx))
    a1_idx = max(1, min(len(a1_array)-3, a1_idx))
    a2_idx = max(1, min(len(a2_array)-3, a2_idx))
    
    # Get local grids
    q_local = q_array[q_idx-1:q_idx+3]
    m_local = mtot_array[m_idx-1:m_idx+3]
    a1_local = a1_array[a1_idx-1:a1_idx+3]
    a2_local = a2_array[a2_idx-1:a2_idx+3]
    
    # Compute weights once
    w_q = lagrange_weight_4pt(q_new, q_local)
    w_m = lagrange_weight_4pt(mtot_new, m_local)
    w_a1 = lagrange_weight_4pt(a1_new, a1_local)
    w_a2 = lagrange_weight_4pt(a2_new, a2_local)
    
    # Tensor product evaluation
    result = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    result += (w_q[i] * w_m[j] * w_a1[k] * w_a2[l] * 
                            data_4d[q_idx-1+i, m_idx-1+j, a1_idx-1+k, a2_idx-1+l])
    
    return result

@njit
def batched_tensor_product_4d(q_array, mtot_array, a1_array, a2_array, data_4d,
                             q_batch, mtot_batch, a1_batch, a2_batch):
    """
    Batched version for multiple interpolation points.
    """
    n = len(q_batch)
    results = np.zeros(n)
    
    for i in range(n):
        results[i] = tensor_product_4d_lagrange(
            q_array, mtot_array, a1_array, a2_array, data_4d,
            q_batch[i], mtot_batch[i], a1_batch[i], a2_batch[i]
        )
    
    return results
