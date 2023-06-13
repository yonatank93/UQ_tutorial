import numpy as np
import emcee


def autocorr(chains, time_axis=1, decorrelate=False, **kwargs):
    """Use ``emcee.autocorr.integrated_time`` to estimate the autocorrelation
    length.

    Parameters
    ----------
    chains: np.ndarray (L, M, N,)
        An array containing multiple chains from multi-chain MCMC simulation.
        The shape of the array should be (nwalkers, nsteps, ndim).
    time_axis: int (Optional)
        Position of the time axis in the chains array. Typically, this is set to
        1, but for emcee, as an example, the chains output sometimes has the
        iterations on axis 0.
    decorrelate: bool (Optional)
        A flag wheter to transform the chains to reduce correlation between
        parameters prior to computing the autocorrelation length. See
        `autocorrelation.decorrelate_chains` on how we reduce the correlation.
    **kwargs: dict
        Keyword arguments for ``emcee.autocorr.integrated_time``.

    Returns
    -------
    np.ndarray (N,)
        Estimated autocorrelation length for each parameter.
    """

    if time_axis == 1:
        # Swap the axes of the chains, so that the time axis is on the first
        # axis (this is the convention in emcee)
        chains = np.swapaxes(chains, 0, 1)

    if decorrelate:
        # Transform the chains to reduce correlation between parameters
        chains = decorrelate_chains(chains)

    return emcee.autocorr.integrated_time(chains, **kwargs)


def decorrelate_chains(chains):
    """Transform the chains (shift and rotate) so that the resulting parameters
    are approximately independent to each other. This is done by first shifting
    the cloud of samples so that the mean coincide with the origin. Then, we
    compute the covaiance matrix of the samples and rotate the cloud of samples
    to align with the eigenvectors of the covariance matrix.

    Parameters
    ----------
    chains: np.ndarray (M, L, N,)
        An array containing multiple chains from multi-chain MCMC simulation.
        The shape of the array should be (nsteps, nwalkers, ndim).

    Returns
    -------
    chain_rotated (M, L, N)
        This is the transformed chains that have been decorrelated.
    """
    _, _, ndim = chains.shape
    # Combine the chains
    chain_combined = chains.reshape((-1, ndim))
    # Compute the mean to shift the center of the cloud of samples to the
    # origin.
    mean = np.mean(chain_combined, axis=0)
    chains_shifted = chains - mean
    # Compute covariance of the samples. We use the covariance to rotate the
    # samples to minimize correlation between parameters.
    cov = np.cov(chain_combined.T)
    # Get the eigenvectors of the covariance matrix
    _, v = np.linalg.eigh(cov)
    # Use the eigenvectors to rotate the samples so that the transformed
    # parameters are approximately independent to each other.
    chain_rotated = chains_shifted @ v

    return chain_rotated
