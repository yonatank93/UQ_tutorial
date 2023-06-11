import numpy as np


def _standard_error_squared(chain: np.ndarray) -> float:
    """Compute the square of the standard error."""
    nn = len(chain)
    se2 = np.var(chain) / nn
    return se2


def mser(
    chain: np.ndarray,
    dmin: int = 0,
    dstep: int = 10,
    dmax: int = -1,
    full_output: bool = False,
) -> int:
    """Estimate the equilibration time using marginal standard error rule
    (MSER). This is done by calculating the standard error (square) of chain_d,
    where chain_d contains the last n-d element of the chain, for progresively
    larger d values, starting from dmin, incremented by dstep. The SE values
    are stored in a list. Then we search the minimum element in the list and
    return the index of that element. To speed up the process, window width can
    be specified. The chain will be redefined to be the mean of the elements in
    every window.

    Parameters
    ----------
    chain: 1D np.ndarray
        Array containing the time series.
    dmin: int
        Index where to start the search in the time series.
    dstep: int
        How much to increment the search is done.
    dmax: int
        Index where to stop the search in the time series.
    full_output: bool
        A flag to return the list of squared standard error.

    Returns
    -------
    dstar: int or dict
        Estimate of the equilibration time using MSER. If ``full_output=True``,
        then a dictionary containing the estimated equilibration time and the
        list of squared standard errors will be returned.
    """
    length = len(chain)

    # Compute the SE square
    SE2_list = [
        _standard_error_squared(chain[dd:])
        for dd in range(length)[dmin:dmax:dstep]
    ]

    # Get the estimate of the equilibration time, wrt the original time series
    dtemp = np.argmin(SE2_list)
    dstar = min([dmin + (dtemp + 1) * dstep, length])

    if full_output:
        return {"dstar": dstar, "SE2": SE2_list}
    else:
        return dstar
