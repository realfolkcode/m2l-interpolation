import numpy as np
from scipy.interpolate import interp1d


def interpolate_latents(X_latent: np.ndarray,
                        Y_latent: np.ndarray,
                        P: np.ndarray,
                        alpha: float) -> np.ndarray:
    """Interpolates two latent signals.

    Args:
        X_latent: Embedding of shape (D, N).
        Y_latent: Embedding of shape (D, M).
        P: The warping path of shape (num_edges, 2).
        alpha: The interpolation coefficient.
    
    Returns:
        Embedding of shape (D, K).
    """
    assert X_latent.shape[-1] <= Y_latent.shape[-1]

    indices = (1 - alpha) * P[:, 0] + alpha * P[:, 1]
    Z = (1 - alpha) * X_latent[:, P[:, 0]] + alpha * Y_latent[:, P[:, 1]]
    
    K = int((1 - alpha) * X_latent.shape[-1] + alpha * Y_latent.shape[-1])
    new_indices =  np.arange(K)

    Z = interp1d(indices, Z, kind='nearest')(new_indices)
    return Z
