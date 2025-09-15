import numpy as np
from scipy.stats import norm
from scipy.special import ndtr  # This is equivalent to the CDF of the normal distribution (Phi)
import torch
import gpytorch


# The module relies on numpy and SciPy's normal distribution utilities.
import numpy as np
from scipy.stats import norm


def exipsi_vectorized(a, b, m, s):
    """
    Vectorized Ψ function for EHVI.
    Parameters:
        a, b (array): Bounds of the intervals (can be scalars or arrays).
        m (array): Means of the Gaussian (can be scalars or arrays).
        s (array): Standard deviations of the Gaussian (can be scalars or arrays).
    Returns:
        array: Ψ values for each input.
    """
    x = s * norm.pdf((b - m) / s) + (a - m) * norm.cdf((b - m) / s)
    return x


def reference_point_on_IN_line(

    pareto_Y: np.ndarray,
    front: str,
    k: float = 1.0,
    cap_frac: float = 0.2,
    tau_fixed: float = 0.05,
):
    """Compute a reference point slightly beyond the current Pareto front.

    Parameters
    ----------
    pareto_Y : np.ndarray
        Current Pareto front in original objective space.
    front : str
        Either ``"upper"`` for maximization or ``"lower"`` for minimization.
    k : float, optional
        Scale for uncertainty-based step.
    cap_frac : float, optional
        Maximum fraction of the ideal–nadir distance to step.
    tau_fixed : float, optional
        Fixed step fraction when uncertainty is not used.

    Returns
    -------
    tuple
        Reference point ``R`` and auxiliary data ``(I, N, u, L)`` describing
        the ideal and nadir points and step geometry.
    """


    Y = np.asarray(pareto_Y, float)

    # Ideal / Nadir in ORIGINAL space
    if front == 'upper':
        I = np.max(Y, axis=0); N = np.min(Y, axis=0)  # max–max
        sign_dir = -1.0  # move "down" past N
    elif front == 'lower':
        I = np.min(Y, axis=0); N = np.max(Y, axis=0)  # min–min
        sign_dir = +1.0  # move "up" past N
    else:
        raise ValueError("This helper assumes both objectives have the same sense.")

    # component-wise distance vector from ideal to nadir point
    d = I - N
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        # Degenerate front: pick a tiny step along first axis
        u = np.array([1.0, 0.0]); L = 1.0
    else:
        # unit vector in the direction from N to I
        u = d / L

    # Step size s
    s = float(np.clip((tau_fixed or 0.1) * L, 0.0, cap_frac * L))

    # Reference point on the ray through N, away from I
    R = N + sign_dir * s * u

    # ensure strict domination vs current front 
    if front == 'upper':
        # R must be <= N componentwise (and strictly < in at least one dim)
        R = np.minimum(R, N - 1e-12)
    else:
        # R must be >= N componentwise (and strictly > in at least one dim)
        R = np.maximum(R, N + 1e-12)

    return R, I, N, u, L


def front_augmentation(pareto_front, front):

    """Augment and orient the Pareto front for EHVI computation.

    Parameters
    ----------
    pareto_front : array-like
        Set of non-dominated objective pairs.
    front : str
        ``"upper"`` for max–max fronts, ``"lower"`` for min–min.
    epsilons : tuple, optional
        Optional epsilon adjustments for each objective.

    Returns
    -------
    np.ndarray
        Augmented front including reference points suitable for EHVI.
    """


    minB = np.min(pareto_front[:,0])
    minD = np.min(pareto_front[:,1])

    maxB = np.max(pareto_front[:,0])
    maxD = np.max(pareto_front[:,1])

    if front =='upper':
        rB = minB - 0.5*np.abs(minB)
        rD = minD - 0.5*np.abs(minD)
        R, I, N, u, L = reference_point_on_IN_line(pareto_front, front='upper', tau_fixed=0.01)
    elif front == 'lower':
        rB = maxB + 0.5*np.abs(maxB)
        rD = maxD + 0.5*np.abs(maxD)
        R, I, N, u, L = reference_point_on_IN_line(pareto_front, front='lower', tau_fixed=0.01)
    else:
        raise ValueError("Invalid front type. Choose 'upper' or 'lower'.") 
    reference_point = [R[0], R[1]]
    #reference_point = [rB, rD]

    # add back epsilons 

    # if epsilons is not None:
    #     reference_point[0] = reference_point[0] + epsilons[0]
    #     reference_point[1] = reference_point[1] + epsilons[1]
    
    if front == 'upper':
        pareto_front = [(-y1, -y2) for y1, y2 in pareto_front]
        reference_point = tuple(-x for x in reference_point) # flip for upper front since we do minimization
    elif front == 'lower': 
        pareto_front = [(y1, y2) for y1, y2 in pareto_front] # no change needef for lower front 
    else:   
        raise ValueError("Invalid front type. Choose 'upper' or 'lower'.")
    # Augment the Pareto front
    # reference_point = tuple(-x for x in reference_point)
    r1, r2 = reference_point
    if front == 'upper':
        augmented_front = np.array([(r1, -1e10)] + sorted(pareto_front, key=lambda p: p[1]) + [(-1e10, r2)])
    elif front == 'lower':
        augmented_front = np.array([(r1, -1e10)] + sorted(pareto_front, key=lambda p: p[1]) + [(-1e10, r2)])

    return augmented_front


def ehvi_maximization(mus1, sigmas1, mus2, sigmas2, augmented_front):
    """
    EHVI calculation for minimization problems for batches of predicted points. Technically this expects a Pareto front to be minimized.
    
    Parameters:
        mus1 (array): Mean vector for the first objective, shape (Nsamples,).
        sigmas1 (array): Standard deviation vector for the first objective, shape (Nsamples,).
        mus2 (array): Mean vector for the second objective, shape (Nsamples,).
        sigmas2 (array): Standard deviation vector for the second objective, shape (Nsamples,).
        augmented_front (array): Augmented Pareto front as a numpy array, shape (Nfront+2, 2).
    
    Returns:
        array: EHVI values for each predicted point in the batch, shape (Nsamples,).
    """
    # Extract coordinates from the augmented Pareto front
    y1 = augmented_front[:, 0]
    y2 = augmented_front[:, 1]
    
    # Define previous and current coordinates for stripes
    y1_prev = y1[:-1]  # Shape (Nfront+1,)
    y1_curr = y1[1:]   # Shape (Nfront+1,)
    y2_prev = y2[:-1]  # Shape (Nfront+1,)
    y2_curr = y2[1:]   # Shape (Nfront+1,)

    # Reshape y1 and y2 to broadcast with the batch (Nsamples,)
    y1_prev = y1_prev[:, np.newaxis]  # Shape (Nfront+1, 1)
    y1_curr = y1_curr[:, np.newaxis]  # Shape (Nfront+1, 1)
    y2_prev = y2_prev[:, np.newaxis]  # Shape (Nfront+1, 1)
    y2_curr = y2_curr[:, np.newaxis]  # Shape (Nfront+1, 1)

    # Reshape mus and sigmas to broadcast with the Pareto front (Nfront+1,)
    mus1 = mus1[np.newaxis, :]  # Shape (1, Nsamples)
    sigmas1 = sigmas1[np.newaxis, :]  # Shape (1, Nsamples)
    mus2 = mus2[np.newaxis, :]  # Shape (1, Nsamples)
    sigmas2 = sigmas2[np.newaxis, :]  # Shape (1, Nsamples)

    # Compute term1 (rectangular contribution)
    term1 = (y1_prev - y1_curr) * norm.cdf((y1_curr - mus1) / sigmas1) \
            * exipsi_vectorized(y2_curr, y2_curr, mus2, sigmas2)
    
    # Compute term2 (difference in Ψ contributions)
    psi_term_prev = exipsi_vectorized(y1_prev, y1_prev, mus1, sigmas1)
    psi_term_curr = exipsi_vectorized(y1_prev, y1_curr, mus1, sigmas1)
    term2 = (psi_term_prev - psi_term_curr) * exipsi_vectorized(y2_curr, y2_curr, mus2, sigmas2)
    
    # Combine terms across stripes (sum over axis 0)
    ehvi_values = np.sum(term1 + term2, axis=0)  # Shape (Nsamples,)
    
    return ehvi_values


