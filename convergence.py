import numpy as np


def rhat(chain, time_axis=1, return_WB=False):
    """Compute the value of :math:`\\hat{r}` proposed by Brooks and Gelman
    (1998). If the samples come from PTMCMC simulation, then the chain needs
    to be from one of the temperature only.

    Parameters
    ----------
    chain: ndarray
        The MCMC chain as a ndarray, preferrably with the shape
        (nwalkers, nsteps, ndims). However, the shape can also be
        (nsteps, nwalkers, ndims), but the argument time_axis needs to be set
        to 0.
    time_axis: int (optional)
        Axis in which the time series is stored (0 or 1). For emcee results,
        the time series is stored in axis 0, but for ptemcee for a given
        temperature, the time axis is 1.
    return_WB: bool (optional)
        A flag to return covariance matrices within and between chains.

    Returns
    -------
    r: float
        Value of PSRF.
    W, B: 2d ndarray
        Matrices of covariance within and between the chains.
    """
    if not time_axis:
        # Reshape the chain so that the time axis is in axis 1
        temp = _reshape_chain(chain)
        chain = temp

    m, n, _ = chain.shape
    lambda1, W, B = _lambda1(chain)
    r = 1 - 1 / n + (1 + 1 / m) * lambda1

    if return_WB:
        toreturn = (r, W, B)
    else:
        toreturn = r

    return toreturn


def _reshape_chain(chain):
    """Reshape the chain and make so that the time series is in axis 1."""
    reshaped_chain = np.swapaxes(chain, 0, 1)
    return reshaped_chain


def _B_over_n(chain):
    """Compute covariance matrix between the chains."""
    return np.cov(np.mean(chain, axis=1), rowvar=False, ddof=1)


def _W(chain):
    """Compute the mean of the covariance matrix within each chain."""
    m, n, nparams = chain.shape
    Wm = np.empty((m, nparams, nparams))
    for walker in range(m):
        Wm[walker] = np.cov((chain[walker]), rowvar=False, ddof=1)
    return np.mean(Wm, axis=0)


def _lambda1(chain):
    """Compute the largest eigenvalue of :math:`W^{-1} B/n`."""
    W = _W(chain)
    B_over_n = _B_over_n(chain)
    V = np.linalg.lstsq(W, B_over_n, rcond=-1)[0]
    # # Force V to be symmetric, which it should be
    # V = (V.T + V) / 2
    # s = np.linalg.eigvalsh(V)
    s = np.linalg.svd(V, compute_uv=False)
    return np.max(s), W, B_over_n
