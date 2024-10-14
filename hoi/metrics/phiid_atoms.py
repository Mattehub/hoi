from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_it
from hoi.core.mi import get_mi, compute_mi_comb, compute_mi_comb_phi
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(1, 2, 3))
def compute_phiid_atoms(inputs, comb, mi_fcn_r=None, mi_fcn=None):
    x, y, ind, ind_red, atom = inputs

    n_var = x.shape[0]

    # select combination
    x_c = x[:, comb, :]
    y_c = y[:, comb, :]

    # compute max(I(x_{-j}; S))
    _, i_minj = jax.lax.scan(mi_fcn_r, (x_c, y_c), ind_red)

    _, i_tot = mi_fcn((x, y_c), comb)

    # compute max(I(x_{-j}; S))
    _, i_maxj_forward = jax.lax.scan(mi_fcn, (x_c, y_c), ind)
    _, i_maxj_backward = jax.lax.scan(mi_fcn, (y_c, x_c), ind)

    rtr = i_minj.min(0)
    r01tx = jnp.minimum(i_minj[0, :], i_minj[2, :])
    r01ty = jnp.minimum(i_minj[1, :], i_minj[3, :])
    r01txy = i_maxj_backward.min(0)
    rxyt0 = jnp.minimum(i_minj[0, :], i_minj[1, :])
    rxyt1 = jnp.minimum(i_minj[2, :], i_minj[3, :])
    rxyt01 = i_maxj_forward.min(0)
    I0tx = i_minj[0, :]
    I0ty = i_minj[1, :]
    I1tx = i_minj[2, :]
    I1ty = i_minj[3, :]
    I01tx = i_maxj_backward[1, :]
    I01ty = i_maxj_backward[0, :]
    Ixyt0 = i_maxj_forward[1, :]
    Ixyt1 = i_maxj_forward[0, :]
    I01txy = i_tot

    knowns_to_atoms_mat = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # rtr
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxyta
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytb
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytab
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rabtx
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Rabty
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Rabtxy
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixta
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixtb
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Iyta
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # Iytb
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # Ixyta
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Ixytb
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixtab
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # Iytab
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ixytab
    ]

    knowns_to_atoms_mat = jnp.array(knowns_to_atoms_mat)
    knowns_to_atoms_mat_multd = jnp.tile(
        knowns_to_atoms_mat[jnp.newaxis, :, :], (n_var, 1, 1)
    )
    b = jnp.concatenate(
        (
            rtr[jnp.newaxis, :],
            r01tx[jnp.newaxis, :],
            r01ty[jnp.newaxis, :],
            r01txy[jnp.newaxis, :],
            rxyt0[jnp.newaxis, :],
            rxyt1[jnp.newaxis, :],
            rxyt01[jnp.newaxis, :],
            I0tx[jnp.newaxis, :],
            I0ty[jnp.newaxis, :],
            I1tx[jnp.newaxis, :],
            I1ty[jnp.newaxis, :],
            I01tx[jnp.newaxis, :],
            I01ty[jnp.newaxis, :],
            Ixyt0[jnp.newaxis, :],
            Ixyt1[jnp.newaxis, :],
            I01txy[jnp.newaxis, :],
        ),
        axis=1,
    )

    out = jnp.linalg.solve(knowns_to_atoms_mat_multd, b)

    return inputs, out[:, atom]


class atoms_phiID(HOIEstimator):
    r"""Integrated Information Decomposition (phiID).

    For each couple of variable the phiID is performed,
    using the Minimum Mutual Information (MMI) approach:

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1), (2, 7)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Luppi et al, 2022 :cite:`luppi2022synergistic`
    """

    __name__ = "Synergy phiID MMI"
    _encoding = False
    _positive = "synergy"
    _negative = "null"
    _symmetric = False

    def __init__(self, x, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, multiplets=multiplets, verbose=verbose
        )

    def fit(
        self,
        minsize=2,
        tau=1,
        direction_axis=0,
        maxsize=None,
        method="gc",
        samples=None,
        atom=15,
        **kwargs,
    ):
        r"""Integrated Information Decomposition (phiID).

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gc', 'binning', 'knn', 'kernel', callable}
            Name of the method to compute entropy. Use either :

                * 'gc': gaussian copula entropy [default]. See
                  :func:`hoi.core.entropy_gc`
                * 'gauss': gaussian entropy. See :func:`hoi.core.entropy_gauss`
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized. See
                  :func:`hoi.core.entropy_bin`
                * 'knn': k-nearest neighbor estimator. See
                  :func:`hoi.core.entropy_knn`
                * 'kernel': kernel-based estimator of entropy
                  see :func:`hoi.core.entropy_kernel`
                * A custom entropy estimator can be provided. It should be a
                  callable function written with Jax taking a single 2D input
                  of shape (n_features, n_samples) and returning a float.

        samples : np.ndarray
            List of samples to use to compute HOI. If None, all samples are
            going to be used.
        tau : int | 1
            The length of the delay to use to compute the redundancy as
            defined in the phiID.
            Default 1
        direction_axis : {0,2}
            The axis on which to consider the evolution,
            0 for the samples axis, 2 for the variables axis.
            Default 0
        atom : {0, 1, 2, ..., 15}
            The information atoms in the phiID to give as output.
        kwargs : dict | {}
            Additional arguments are sent to each MI function

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-rder interactions of
            shape (n_multiplets, n_variables)
        """
        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing mi
        x, kwargs = prepare_for_it(self._x, method, samples=samples, **kwargs)

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(compute_mi_comb, mi=mi_fcn)
        compute_mi_r = partial(compute_mi_comb_phi, mi=mi_fcn)
        compute_at = partial(
            compute_phiid_atoms, mi_fcn_r=compute_mi_r, mi_fcn=compute_mi
        )

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________

        offset = 0
        if direction_axis == 2:
            hoi = jnp.zeros(
                (len(order), self.n_variables - tau), dtype=jnp.float32
            )
        else:
            hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)

        for msize in pbar:
            pbar.set_description(
                desc="SynPhiIDMMI order %s" % msize, refresh=False
            )

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # define indices for I(x_{-j}; S)
            ind = (jnp.mgrid[0:msize, 0:msize].sum(0) % msize)[:, 1:]

            dd = jnp.array(np.meshgrid(jnp.arange(msize), jnp.arange(msize))).T
            ind_red = dd.reshape(-1, 2, 1)

            if direction_axis == 0:
                x_c = x[:, :, :-tau]
                y = x[:, :, tau:]

            elif direction_axis == 2:
                x_c = x[:-tau, :, :]
                y = x[tau:, :, :]

            else:
                raise ValueError("axis can be eaither equal 0 or 2.")

            # compute hoi
            _, _hoi = jax.lax.scan(
                compute_at, (x_c, y, ind, ind_red, atom), _h_idx
            )

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
