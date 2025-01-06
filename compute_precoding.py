# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
from scipy.linalg import pinvh, inv


def compute_precoding_simple(interference_estimation: np.ndarray) -> np.ndarray:
    """Compute the digital self-interference precoding matrix using a simple approach.

    Args:

    interference_estimation:
        A 3D numpy array of shape (M, N, T),
        where M is the number of recive antennas,
        N is the number of transmit antennas,
        and T is the number of frequency bins.

    Returns:
        A 2D numpy array of shape (N, T) representing the precoding matrix.
    """

    # Extract parameters from the interference estimation matrix' dimensions
    num_samples = interference_estimation.shape[2]
    num_transmitters = interference_estimation.shape[1]
    num_receivers = interference_estimation.shape[0]

    # Compute the channel diagonal matrix, denoted by \tilde{H}
    channel_diagonals = np.zeros((num_samples * num_receivers, num_samples * num_transmitters), dtype=np.complex_)
    for rx, tx in np.ndindex(num_receivers, num_transmitters):
        channel_diagonals[rx*num_samples:(rx+1)*num_samples, tx*num_samples:(tx+1)*num_samples] = np.diag(interference_estimation[rx, tx, :])

    # Define the optimization problem
    HH = channel_diagonals.T.conj() @ channel_diagonals
    p = cp.Variable(num_samples * num_transmitters, complex=True)
    objective = cp.Minimize(cp.quad_form(p, HH, assume_PSD=False))
    problem = cp.Problem(objective, [cp.norm(p, 2) <= 1, cp.sum(cp.real(p)) == 1, cp.sum(cp.imag(p)) == 1])
    problem.solve()

    precodings = p.value.reshape((num_transmitters, num_samples))
    return precodings

def compute_precoding_frequency(interference_estimation: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Compute the digital self-interference precoding matrix using a simple approach.

    Args:

    interference_estimation:
        A 3D numpy array of shape (M, N, T),
        where M is the number of recive antennas,
        N is the number of transmit antennas,
        and T is the number of frequency bins.

    Returns:
        A 2D numpy array of shape (N, T) representing the precoding matrix.
    """
    
    # Extract parameters from the interference estimation matrix' dimensions
    num_samples = interference_estimation.shape[2]
    num_transmitters = interference_estimation.shape[1]

    precodings = np.zeros((num_transmitters, num_samples), dtype=np.complex_)
    for l in range(num_samples):
        interference_bin = interference_estimation[:, :, l]
        #interference_cov_inv = np.linalg.pinv(interference_bin.T.conj() @ interference_bin)
        interference_cov_inv = pinvh(interference_bin.T.conj() @ interference_bin, return_rank=False)
        precodings[:, l] = interference_cov_inv @ a / (a.T @ interference_cov_inv @ a)
        
    # Correct the amplitude of the precoding matrix
    precodings *= num_transmitters

    return precodings


def compute_precoding_frequency_regularized(
    interference_estimation: np.ndarray,
    a: np.ndarray,
    regularization: float = 0.0,
) -> np.ndarray:
    """Compute the digital self-interference precoding matrix using a simple approach.

    Args:

    interference_estimation:
        A 3D numpy array of shape (M, N, T),
        where M is the number of recive antennas,
        N is the number of transmit antennas,
        and T is the number of frequency bins.

    Returns:
        A 2D numpy array of shape (N, T) representing the precoding matrix.
    """
    
    # Extract parameters from the interference estimation matrix' dimensions
    num_samples = interference_estimation.shape[2]
    num_transmitters = interference_estimation.shape[1]
    num_receivers = interference_estimation.shape[0]

    precodings = np.zeros((num_transmitters, num_samples), dtype=np.complex_)
    for l in range(num_samples):
        interference_bin = interference_estimation[:, :, l]
        interference_cov_inv = pinvh(
            interference_bin.T.conj() @ interference_bin + np.eye(num_transmitters) * regularization, 
            return_rank=False,
        )
        precodings[:, l] = interference_cov_inv @ a / (a.T @ interference_cov_inv @ a)
        
    # Correct the amplitude of the precoding matrix
    precodings *= num_transmitters

    return precodings


def compute_precoding_scalar(interference_estimation: np.ndarray, a: np.ndarray) -> np.ndarray:
    summed_channel = np.sum(interference_estimation, axis=2, keepdims=False)
    inverse = pinvh(summed_channel.T.conj() @ summed_channel, check_finite=False)
    scalar_precoding = inverse @ a / (a.T @ inverse @ a)
    
    return scalar_precoding
