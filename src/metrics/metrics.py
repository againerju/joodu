import numpy as np
import math


def log_likelihood(
        y: np.ndarray, 
        y_hat: np.ndarray,
        pi: np.ndarray, 
        sigma: np.ndarray
        ) -> np.ndarray:
    """
    Compute the Gaussian mixture log-likelihood.

    Args:
        y: ground truth trajectory [T, 2]
        y_hat: predicted trajectories [K, T, 2]
        pi: confidence weights associated with trajectories [K]
        sigma: distribution standart deviation [K, T]
    Return:
        log-likelihood
    """
    displacement_norms_squared = np.sum(((y - y_hat)) ** 2 , axis=-1)

    if isinstance(sigma, np.ndarray):
        normalizing_const = np.log(2. * math.pi * sigma ** 2)
        lse_args = np.log(pi) - np.sum(normalizing_const + np.divide(0.5 * displacement_norms_squared, sigma**2), axis=-1)
    
    else:
        sigma = 1.0
        normalizing_const = np.log(2. * math.pi * sigma ** 2)
        lse_args = np.log(pi) - np.sum(normalizing_const + 0.5 * displacement_norms_squared / sigma ** 2, axis=-1)

    max_arg = lse_args.max()
    ll = np.log(np.sum(np.exp(lse_args - max_arg))) + max_arg

    return ll
