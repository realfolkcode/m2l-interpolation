import numpy as np
import libfmp.c3


def compute_frechet_function(X: np.ndarray,
                             Y: np.ndarray,
                             Z: np.ndarray) -> float:
    """Computes the Frechet function value for embedding Z.
    
    Args:
        X: Embedding of shape (D, N).
        Y: Embedding of shape (D, M).
        Z: Embedding of shape (D, K).

    Returns:
        The normalized Frechet function for embedding Z (divided by K).
    """
    C = libfmp.c3.compute_cost_matrix(Z, X)
    D = libfmp.c3.compute_accumulated_cost_matrix(C)
    dist_X = D[-1, -1] / Z.shape[1]

    C = libfmp.c3.compute_cost_matrix(Z, Y)
    D = libfmp.c3.compute_accumulated_cost_matrix(C)
    dist_Y = D[-1, -1] / Z.shape[1]
    
    return (dist_X**2 + dist_Y**2) / 2
