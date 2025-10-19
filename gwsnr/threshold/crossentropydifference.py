import numpy as np
from scipy.stats import gaussian_kde

# function for cross-entropy
# kde cut needs to be defined outside the function
def cross_entropy(kde_detected, kde_with_cut, sample_size=10000):
    # sample from the kde detected
    samples = kde_detected.resample(sample_size)
    # check if samples is 1D or multi-D
    # print(f'samples.shape: {samples.shape}')

    # evaluate the kde with cut on the samples
    p_hat = kde_with_cut(samples)
    p_hat_true = kde_detected(samples)
    # avoid log(0) by setting a minimum value
    p_hat = np.clip(p_hat, np.finfo(np.float64).tiny, None)
    p_hat_true = np.clip(p_hat_true, np.finfo(np.float64).tiny, None)
    # compute the cross-entropy
    H = np.sum(np.log(p_hat))
    H_true = np.sum(np.log(p_hat_true))
    
    return H, H_true

# maximize the cross-entropy by varying the snr threshold
# from scipy.optimize import minimize_scalar
def cross_entropy_difference(input_args):

    snr_thr=input_args[0]
    size=input_args[1]
    false_alarm_rate =input_args[2]
    false_alarm_rate_cut = input_args[3]
    observed_snr_net=input_args[4]
    parameters_to_fit=input_args[5]
    iter=input_args[6]

    # define a new kde with the snr cut
    snr_cut_idx = observed_snr_net >= snr_thr
    # check number of dimensions
    dim = len(parameters_to_fit.shape)
    
    if dim < 2:
        # 1D case
        parameters_to_fit_cut = parameters_to_fit[snr_cut_idx]
        
        # define the kde for the detected injections
        far_cut_idx = false_alarm_rate < false_alarm_rate_cut
        parameters_detected = parameters_to_fit[far_cut_idx]
        
        len_ = len(parameters_to_fit_cut)
        size = min(size, len_) if size is not None else len_

        parameters_to_fit_cut = np.random.choice(parameters_to_fit_cut, size=size, replace=False)
        len_ = len(parameters_detected)
        size = min(size, len_) if size is not None else len_
        parameters_detected = np.random.choice(parameters_detected, size=size, replace=False)

        kde_with_cut = gaussian_kde(parameters_to_fit_cut)
        kde_detected = gaussian_kde(parameters_detected)
        
    else:
        # Multi-dimensional case
        parameters_to_fit_cut = []
        for i in range(dim):
            parameters_to_fit_cut.append(parameters_to_fit[i][snr_cut_idx])
        parameters_to_fit_cut = np.array(parameters_to_fit_cut)
        
        # define the kde for the detected injections
        far_cut_idx = false_alarm_rate < false_alarm_rate_cut
        parameters_detected = []
        for i in range(dim):
            parameters_detected.append(parameters_to_fit[i][far_cut_idx])
        
        parameters_detected = np.array(parameters_detected)
        
        len_ = len(parameters_to_fit_cut[0])
        size1 = min(size, len_) if size is not None else len_
        len_ = len(parameters_detected[0])
        size2 = min(size, len_) if size is not None else len_
        parameters_to_fit_cut_ = []
        parameters_detected_ = []
        for i in range(dim):
            parameters_to_fit_cut_.append(np.random.choice(len(parameters_to_fit_cut[0]), size=size1, replace=False))

            parameters_detected_.append(np.random.choice(len(parameters_detected[0]), size=size2, replace=False))

        kde_with_cut = gaussian_kde(parameters_to_fit_cut_)
        kde_detected = gaussian_kde(parameters_detected_)

    # compute the negative cross-entropy (since we want to maximize it)
    H, H_true = cross_entropy(kde_detected, kde_with_cut)
    del_H = H - H_true

    # # print shape of all the outputs
    # print(f"Output shapes - del_H: {del_H_i.shape}, H: {H_i.shape}, H_true: {H_true_i.shape}, iter: {iter_i.shape}")

    return del_H, H, H_true, iter