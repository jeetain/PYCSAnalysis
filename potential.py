"""
Contains various pairwise potential models.
"""

import numpy
import scipy.optimize

def lj(r, sigma, epsilon):
    """
    Represents a 12-6 Lennard-Jones potential.

    Parameters
    ----------
    r : ndarray
        Separation distances
    sigma : float
        Length scale factor
    epsilon : float
        Energy scale factor
    
    Returns
    -------
    ndarray
        Potential energies
    """

    r_inv = sigma / r
    return 4 * epsilon * ((r_inv ** 12) - (r_inv ** 6))

def lj_lambda(r, sigma, epsilon, lambda_):
    """
    Represents an "LJ-lambda" potential.

    Parameters
    ----------
    r : ndarray
        Separation distances
    sigma : float
        Length scale factor
    epsilon : float
        Energy scale factor
    lambda_ : float
        Potential shape factor

    Returns
    -------
    ndarray
        Potential energies
    """

    r_min = (2 ** (1 / 6)) * sigma
    r_inv = sigma / r
    u_lj = 4 * epsilon * ((r_inv ** 12) - (r_inv ** 6))
    u_rep = u_lj + epsilon
    u_rep[r > r_min] = 0
    return u_rep + (lambda_ * (u_lj - u_rep))

def lj_piecewise_full(r, sigma, epsilon, m, n, s):
    """
    Represents an attractive-repulsive piecewise modified LJ potential.

    Parameters
    ----------
    r : ndarray
        Separation distances
    sigma : float
        Length scale factor
    epsilon : float
        Energy scale factor
    m : float
        Far (attractive) side exponent
    n : float
        Near (repulsive) side exponent
    s : float
        Horizontal shift factor

    Returns
    -------
    ndarray
        Potential energies
    """
    
    # Precalculate constants
    f_shift = (2 ** (1 / 6)) + s
    f_root_m = 2 ** (1 / m)
    f_root_n = 2 ** (1 / n)
    
    # Calculate terms
    r_nd = r / sigma
    mu = 1 / (r_nd + f_root_m - f_shift)
    nu = 1 / (r_nd + f_root_n - f_shift)
    
    # Calculate energies
    u_nd = numpy.empty_like(r_nd)
    mu_mask = r_nd > f_shift
    nu_mask = ~mu_mask
    u_nd[mu_mask] = 4.0 * ((mu[mu_mask] ** (2 * m)) - (mu[mu_mask] ** m))
    u_nd[nu_mask] = 4.0 * ((nu[nu_mask] ** (2 * n)) - (nu[nu_mask] ** n))

    return u_nd * epsilon

def lj_piecewise_repel(r, sigma, epsilon, n, s):
    """
    Represents a purely repulsive cutoff modified LJ potential.

    Parameters
    ----------
    r : ndarray
        Separation distances
    sigma : float
        Length scale factor
    epsilon : float
        Energy scale factor
    n : float
        Near (repulsive) side exponent
    s : float
        Horizontal shift factor

    Returns
    -------
    ndarray
        Potential energies
    """

    # Precalculate constants
    f_shift = (2 ** (1 / 6)) + s
    f_root_n = 2 ** (1 / n)
    
    # Calculate terms
    r_nd = r / sigma
    nu = 1 / (r_nd + f_root_n - f_shift)
    
    # Calculate energies
    u_nd = numpy.zeros_like(r_nd)
    nu_mask = r_nd < f_shift
    u_nd[nu_mask] = 4.0 * ((nu[nu_mask] ** (2 * n)) - (nu[nu_mask] ** n))
    
    return u_nd * epsilon

def jagla(r, sigma, epsilon, n, s, a0, a1, a2, b0, b1, b2, q):
    """
    Represents a generalized "Jagla" potential.

    Parameters
    ----------
    r : ndarray
        Separation distances
    sigma : float
        Power term length scale factor
    epsilon : float
        Power term energy scale factor
    n : float
        Power term exponent
    s : float
        Power term horizontal shift distance
    a0 : float
        First exponential term energy scale factor
    a1 : float
        First exponential term shape factor
    a2 : float
        First exponential term horizontal shift distance
    b0 : float
        Second exponential term energy scale factor
    b1 : float
        Second exponential term shape factor
    b2 : float
        Second exponential term horizontal shift distance
    q : float
        Overall horizontal scale factor

    Returns
    -------
    ndarray
        Potential energies
    """

    rs = r * q
    return (epsilon * ((sigma / (rs - s)) ** n)) + \
        (a0 / (1 + numpy.exp(a1 * (rs - a2)))) - \
        (b0 / (1 + numpy.exp(b1 * (rs - b2))))

def jagla_solve(e):
    """
    Determines "Jagla" potential coefficients for a specific minimum position.

    Parameters
    ----------
    e : float
        Desired potential well depth

    Returns
    -------
    tuple
        Potential parameters **sigma**, **epsilon**, **n**, **s**, 
        **a0**, **a1**, **a2**, **b0**, **b1**, **b2**, **q**

    Raises
    ------
    RuntimeError
        The minimization algorithm failed
    """

    # Define the fixed parameters for the Jagla system
    sigma, epsilon, n, s, a0, a1, a2, b1, b2 = \
        0.2, 10.0, 36.0, 0.8, 11.0346, 404.396, 1.0174094, 1044.5, 1.0305952

    # Define the solver residual function
    def residual(b0):
        objective = lambda r: jagla(r, sigma, epsilon, n, s, a0, a1, a2, b0, b1, b2, 1.0)
        minimize_result = scipy.optimize.minimize_scalar(objective, method="bounded", \
            bounds=(1.0, 1.05), options={"xatol": 0.0})
        if not minimize_result.success:
            raise RuntimeError("Convergence error during Jagla minimization: {}".format(minimize_result.message))
        return minimize_result.x, minimize_result.fun + e
    
    # Solve for b0 and return the results
    b0 = scipy.optimize.brentq(lambda b0: residual(b0)[1], 0.0, 2.0)
    return sigma, epsilon, n, s, a0, a1, a2, b0, b1, b2, residual(b0)[0]

def stsp(r, sigma, epsilon, lambda_, n, s):
    """
    Represents an "STSP" potential.

    Parameters
    ----------
    r : ndarray
        Separation distances
    sigma : float
        Length scale factor
    epsilon : float
        Energy scale factor
    lambda_ : float
        Potential shape factor
    n : float
        Potential exponent
    s : float
        Horizontal shift factor
    """

    nu = (r / sigma) - s + (2 ** (1 / n))
    return 4 * epsilon * ((nu ** (-2 * n)) - (lambda_ * (nu ** -n)))
