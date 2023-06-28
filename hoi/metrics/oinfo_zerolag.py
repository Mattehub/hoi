from math import comb
import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.core.combinatory import combinations
from hoi.core.entropies import get_entropy, copnorm_nd
from hoi.core.oinfo import oinfo_scan

logger = logging.getLogger("frites")


def oinfo_zerolag(
        data, y=None, minsize=3, maxsize=None, method='gcmi', **kwargs
    ):
    """Dynamic, possibly task-related oinfo.

    Parameters
    ----------
    data : array_like
            Standard NumPy arrays of shape (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info.
    minsize, maxsize : int | 3, 5
        Minimum and maximum size of the multiplets
    method : {'gcmi', 'binning', 'knn'}
        Name of the method to compute entropy. Use either :

            * 'gcmi': gaussian copula entropy [default]
            * 'binning': binning-based estimator of entropy. Note that to use
              this estimator, the data have be to discretized
            * 'knn': k-nearest neighbor estimator

    kwargs : dict | {}
        Additional arguments are sent to each entropy function

    Returns
    -------
    oinfo : array_like
        The O-info array of shape (n_multiplets, n_variables) where positive
        values reflect redundant dominated interactions and negative values
        stand for synergistic dominated interactions.
    """
    # ________________________________ INPUTS _________________________________
    # force data to be 3d
    assert data.ndim >= 2
    if data.ndim == 2:
        data = data[..., np.newaxis]

    # inputs conversion
    is_task_related = isinstance(y, (str, list, np.ndarray, tuple))

    # extract variables
    n_samples, n_features, n_variables = data.shape

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = n_features
    maxsize = max(1, maxsize)
    assert maxsize >= minsize

    logger.info(
        f"Compute the {'task-related ' * is_task_related} HOI "
        f"(min={minsize}; max={maxsize})"
    )

    # ____________________________ PREPROCESSING ______________________________

    # for task-related, add behavior along spatial dimension
    if is_task_related:
        y = np.tile(y.reshape(-1, 1, 1), (1, 1, n_variables))
        data = np.concatenate((data, data), axis=1)

    # method specific preprocessing
    if method == 'gcmi':
        logger.info('    copnorm data')
        data = copnorm_nd(data, axis=0)
        data = data - data.mean(axis=0, keepdims=True)
        kwargs['demean'] = False
    elif method == 'binning':
        if data.dtype != int:
            raise ValueError(
                "data dtype should be integer. Check that you discretized your"
                " data. If so, use `data.astype(int)`"
            )
        n_bins = len(np.unique(data))
        logger.info(f'    {n_bins} bins detected from the data')
        kwargs['n_bins'] = n_bins

    # make the data (n_variables, n_features, n_trials)
    data = jnp.asarray(data.transpose(2, 1, 0))

    # ________________________________ O-INFO _________________________________

    # get the function to compute entropy and vmap it twice for 4D inputs
    entropy = jax.vmap(jax.vmap(
        get_entropy(method=method, **kwargs)
    ))

    # use it to compute oinfo
    oinfo_mmult = jax.jit(partial(oinfo_scan, entropy=entropy))


    oinfo = []
    for msize in range(minsize, maxsize + 1):
        logger.info(f"    Multiplets of size {msize}")
        combs = combinations(n_features, msize, task_related=is_task_related)

        _, _oinfo = jax.lax.scan(oinfo_mmult, data, combs)
        oinfo.append(np.asarray(_oinfo))

    oinfo = np.concatenate(oinfo, axis=0)
    return oinfo


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    import seaborn as sns
    import time as tst

    # from idtxl.idtxl_utils import discretise

    set_mpl_style()

    np.random.seed(0)

    ###########################################################################
    n_trials = 600
    n_roi = 5
    n_times = 50

    redundancy = [
        (2, 3, 4)
    ]
    synergy = [
        (0, 1, 2)
    ]
    ###########################################################################

    def set_redundancy(x, redundancy, sl, win, trials):
        for m in redundancy:
            x[:, m, sl] += .8 * trials.reshape(-1, 1, 1) * win
        return x

    def set_synergy(x, synergy, sl, win, trials):
        for m in synergy:
            blocks = np.array_split(np.arange(n_trials), len(m))
            for n_b, b in enumerate(blocks):
                x[b, m[n_b], sl] += trials[b].reshape(-1, 1) * win[0, ...]
        return x


    # generate the data
    x = np.random.rand(n_trials, n_roi, n_times)
    roi = np.array([f"r{r}" for r in range(n_roi)])[::-1]
    trials = np.random.rand(n_trials)
    times = np.arange(n_times)
    win = np.hanning(10).reshape(1, 1, -1)

    # introduce (redundant, synergistic) information
    x = set_redundancy(x, redundancy, slice(20, 30), win, trials)
    x = set_synergy(x, synergy, slice(30, 40), win, trials)


    start_time = tst.time()

    # for nt in range(x.shape[-1]):
    #     x[:, :, nt] = discretise(x[:, :, nt], 8)
    # x = x.astype(int)


    # x = x[:, :, 50:100]
    oinfo = oinfo_zerolag(x, minsize=3, maxsize=4, method='gcmi')
    print(oinfo)

    vmin, vmax = np.percentile(oinfo, [1, 99])
    minmax = min(abs(vmin), abs(vmax))

    # plt.pcolormesh(oinfo, cmap='RdBu_r', vmin=-minmax, vmax=minmax)
    plt.pcolormesh(oinfo, cmap='RdBu_r')
    plt.colorbar()
    plt.show()
