from __future__ import annotations
import numpy
import numpy as np
from contextlib import contextmanager
from typing import cast
from nbodykit.algorithms.fftpower import _find_unique_edges
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.algorithms.fftpower import project_to_basis
from nbodykit.base.catalog import CatalogSourceBase
from nbodykit.base.mesh import MeshSource
from pmesh.pm import ComplexField
from pmesh.pm import RealField


@contextmanager
def rescale_y3d(
    y3d: ComplexField, *, los=[0, 0, 1], alpara: float = 1.0, alperp: float = 1.0,
):
    rawx = [xx.copy() for xx in y3d.x]
    for xx, is_z in zip(y3d.x, los):
        xx[:] *= alpara if is_z else alperp
    yield y3d
    y3d.x = rawx


def _cast_source(
    source, BoxSize, Nmesh, resampler="cic", interlaced=False, compensated=True,
):
    from pmesh.pm import Field
    from nbodykit.source.mesh import FieldMesh

    if isinstance(source, Field):
        source = FieldMesh(source)
    elif isinstance(source, CatalogSourceBase):
        if not isinstance(source, MeshSource):
            source = source.to_mesh(
                BoxSize=BoxSize,
                Nmesh=Nmesh,
                dtype="f8",
                resampler=resampler,
                interlaced=interlaced,
                compensated=compensated,
            )
    if not isinstance(source, MeshSource):
        raise TypeError(f"Unknow type of source in FFTPowerAP: {type(source)}")
    if BoxSize is not None and any(source.attrs["BoxSize"] != BoxSize):
        raise ValueError("Mismatched BoxSize between __init__ and source.attrs")
    if Nmesh is not None and any(source.attrs["Nmesh"] != Nmesh):
        raise ValueError(
            "Mismatched Nmesh between __init__ and source.attrs;"
            "if trying to re-sample with a different mesh, specify "
            "`Nmesh` as keyword of to_mesh()"
        )

    return source


class FFTPowerAP(FFTPower):
    """Support AP effect via rebinning, ref: https://arxiv.org/abs/2003.08277

    alpara: float
        true / fiducial, parallel to line-of-sight scale factor, by default 1
    alperp: float
        true / fiducial, perpendicular to line-of-sight scale factor, by default 1

    resampler: str
        works only when first and second are instance of CatalogSourceBase, by default 'cic'
    interlaced: bool
        works only when first and second are instance of CatalogSourceBase, by default False
    compensated: bool
        works only when first and second are instance of CatalogSourceBase, by default True
    arnold: bool
        whether to use arnold's function for binning measurement, by default False

    Notes
    -----
    shot noise and power spectrum has been rescaled by 1 / (alperp**2 * alpara)
    """

    def __init__(
        self,
        first,
        mode,
        Nmesh=None,
        BoxSize=None,
        second=None,
        los=[0, 0, 1],
        Nmu=5,
        dk=None,
        kmin=0.0,
        kmax=None,
        poles=[],
        alpara=1.0,
        alperp=1.0,
        resampler="cic",
        interlaced=False,
        compensated=True,
        arnold=False,
    ):
        self.arnold = arnold
        self.alpara, self.alperp = alpara, alperp
        if not (los == [0, 0, 1] or los == [0, 1, 0] or los == [1, 0, 0]):
            raise ValueError("LOS must be [0, 0, 1], [0, 1, 0] or [1, 0, 0].")

        # copied from FFTPower, check first!
        # mode is either '1d' or '2d'
        if mode not in ["1d", "2d"]:
            raise ValueError("`mode` should be either '1d' or '2d'")

        # check los
        if numpy.isscalar(los) or len(los) != 3:
            raise ValueError("line-of-sight ``los`` should be vector with length 3")
        if not numpy.allclose(numpy.einsum("i,i", los, los), 1.0, rtol=1e-5):
            raise ValueError("line-of-sight ``los`` must be a unit vector")
        # ----------------------------------
        # perform cast_source here
        first = _cast_source(
            first,
            Nmesh=Nmesh,
            BoxSize=BoxSize,
            resampler=resampler,
            interlaced=interlaced,
            compensated=compensated,
        )
        if second is not None:
            second = _cast_source(
                second,
                Nmesh=Nmesh,
                BoxSize=BoxSize,
                resampler=resampler,
                interlaced=interlaced,
                compensated=compensated,
            )
        else:
            second = first

        super().__init__(
            first, mode, Nmesh, BoxSize, second, los, Nmu, dk, kmin, kmax, poles,
        )
        self.attrs["shotnoise"] /= alperp ** 2 * alpara

    # override
    def run(self):
        r"""
        Compute the power spectrum in a periodic box, using FFTs.

        Returns 
        -------
        power : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that holds the measured :math:`P(k)` or
            :math:`P(k,\mu)`. It stores the following variables:

            - k :
                the mean value for each ``k`` bin
            - mu : ``mode=2d`` only
                the mean value for each ``mu`` bin
            - power :
                complex array storing the real and imaginary components of the power
            - modes :
                the number of Fourier modes averaged together in each bin

        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic` or ``None``
            a BinnedStatistic object to hold the multipole results
            :math:`P_\ell(k)`; if no multipoles were requested by the user,
            this is ``None``. It stores the following variables:

            - k :
                the mean value for each ``k`` bin
            - power_L :
                complex array storing the real and imaginary components for
                the :math:`\ell=L` multipole
            - modes :
                the number of Fourier modes averaged together in each bin

        power.attrs, poles.attrs : dict
            dictionary of meta-data; in addition to storing the input parameters,
            it includes the following fields computed during the algorithm
            execution:

            - shotnoise : float
                the power Poisson shot noise, equal to :math:`V/N`, where
                :math:`V` is the volume of the box and `N` is the total
                number of objects; if a cross-correlation is computed, this
                will be equal to zero
            - N1 : int
                the total number of objects in the first source
            - N2 : int
                the total number of objects in the second source
        """

        # only need one mu bin if 1d case is requested
        if self.attrs["mode"] == "1d":
            self.attrs["Nmu"] = 1

        # measure the 3D power (y3d is a ComplexField)
        y3d, attrs = self._compute_3d_power(self.first, self.second)
        y3d = cast(ComplexField, y3d)

        # binning in k out to the minimum nyquist frequency
        # (accounting for possibly anisotropic box)
        dk = self.attrs["dk"]
        kmin = self.attrs["kmin"]
        kmax = self.attrs["kmax"]
        if kmax is None:
            kmax = numpy.pi * y3d.Nmesh.min() / y3d.BoxSize.max() + dk / 2

        if dk > 0:
            kedges = numpy.arange(kmin, kmax, dk)
            kcoords = None
        else:
            raise NotImplementedError
            kedges, kcoords = _find_unique_edges(
                y3d.x, 2 * numpy.pi / y3d.BoxSize, kmax, y3d.pm.comm
            )

        # project on to the desired basis
        muedges = numpy.linspace(-1, 1, self.attrs["Nmu"] + 1, endpoint=True)
        edges = [kedges, muedges]
        coords = [kcoords, None]
        with rescale_y3d(
            y3d, los=self.attrs["los"], alpara=self.alpara, alperp=self.alperp
        ) as y3d:
            if not self.arnold:
                result, pole_result = project_to_basis(
                    y3d, edges, poles=self.attrs["poles"], los=self.attrs["los"]
                )
            else:
                result, pole_result = arnold_project_to_basis(
                    y3d,
                    edges,
                    ells=self.attrs["poles"],
                    los=self.attrs["los"],
                    exclude_zero=False,
                )
                # TODO: should keep poles_zero
                pole_result = pole_result[:-1]

        # format the power results into structured array
        if self.attrs["mode"] == "1d":
            cols = ["k", "power", "modes"]
            icols = [0, 2, 3]
            edges = edges[0:1]
            coords = coords[0:1]
        else:
            cols = ["k", "mu", "power", "modes"]
            icols = [0, 1, 2, 3]

        # power results as a structured array
        dtype = numpy.dtype(
            [(name, result[icol].dtype.str) for icol, name in zip(icols, cols)]
        )
        power = numpy.squeeze(numpy.empty(result[0].shape, dtype=dtype))
        for icol, col in zip(icols, cols):
            power[col][:] = numpy.squeeze(result[icol])
        power["power"] /= self.alperp ** 2 * self.alpara

        # multipole results as a structured array
        poles = None
        if pole_result is not None:
            k, poles, N = pole_result
            cols = ["k"] + ["power_%d" % l for l in self.attrs["poles"]] + ["modes"]
            result = [k] + [pole for pole in poles] + [N]

            dtype = numpy.dtype(
                [(name, result[icol].dtype.str) for icol, name in enumerate(cols)]
            )
            poles = numpy.empty(result[0].shape, dtype=dtype)
            for icol, col in enumerate(cols):
                poles[col][:] = result[icol]
            for key in [f"power_{l}" for l in self.attrs["poles"]]:
                poles[key] /= self.alperp ** 2 * self.alpara

        return self._make_datasets(edges, poles, power, coords, attrs)


def arnold_project_to_basis(
    y3d, edges, los=(0, 0, 1), ells=None, antisymmetric=False, exclude_zero=False
):
    r"""
    Project a 3D statistic on to the specified basis. The basis will be one of:

        - 2D :math:`(x, \mu)` bins: :math:`\mu` is the cosine of the angle to the line-of-sight
        - 2D :math:`(x, \ell)` bins: :math:`\ell` is the multipole number, which specifies
          the Legendre polynomial when weighting different :math:`\mu` bins.

    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fftpower.py.

    Notes
    -----
    In single precision (float32/complex64) nbodykit's implementation is fairly imprecise
    due to incorrect binning of :math:`x` and :math:`\mu` modes.
    Here we cast mesh coordinates to the maximum precision of input ``edges``,
    which makes computation much more accurate in single precision.

    Notes
    -----
    We deliberately set to 0 the number of modes beyond Nyquist, as it is unclear whether to count Nyquist as :math:`\mu` or :math:`-\mu`
    (it should probably be half weight for both).
    Our safe choice ensures consistent results between hermitian compressed and their associated uncompressed fields.

    Notes
    -----
    The 2D :math:`(x, \ell)` bins will be computed only if ``ells`` is specified.
    See return types for further details.
    For both :math:`x` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end,
    i.e. mode `mode` falls in bin `i` if ``edges[i] <= mode < edges[i+1]``.
    However, last :math:`\mu`-bin is inclusive on both ends: ``edges[-2] <= mu <= edges[-1]``.
    Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
    Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.

    Warning
    -------
    Integration over Legendre polynomials for multipoles is performed between the first and last :math:`\mu`-edges,
    e.g. with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, integration is performed between :math:`\mu = 0.2` and :math:`\mu = 0.8`.

    Parameters
    ----------
    y3d : RealField or ComplexField
        The 3D array holding the statistic to be projected to the specified basis.

    edges : list of arrays, (2,)
        List of arrays specifying the edges of the desired :math:`x` bins and :math:`\mu` bins; assumed sorted.

    los : array_like, default=(0, 0, 1)
        The line-of-sight direction to use, which :math:`\mu` is defined with respect to.

    ells : tuple of ints, default=None
        If provided, a list of integers specifying multipole numbers to project the 2D :math:`(x, \mu)` bins on to.

    Returns
    -------
    result : tuple
        The 2D binned results; a tuple of ``(xmean2d, mumean2d, y2d, n2d)``, where:

            - xmean2d : array_like, (nx, nmu)
                The mean :math:`x` value in each 2D bin
            - mumean2d : array_like, (nx, nmu)
                The mean :math:`\mu` value in each 2D bin
            - y2d : array_like, (nx, nmu)
                The mean ``y3d`` value in each 2D bin
            - n2d : array_like, (nx, nmu)
                The number of values averaged in each 2D bin

    result_poles : tuple or None
        The multipole results; if ``ells`` supplied it is a tuple of ``(xmean1d, poles, n1d)``,
        where:

            - xmean1d : array_like, (nx,)
                The mean :math:`x` value in each 1D multipole bin
            - poles : array_like, (nell, nx)
                The mean multipoles value in each 1D bin
            - n1d : array_like, (nx,)
                The number of values averaged in each 1D bin
    """
    comm = y3d.pm.comm
    x3d = y3d.x
    hermitian_symmetric = y3d.compressed
    if antisymmetric:
        hermitian_symmetric *= -1

    from scipy.special import legendre

    # Setup the bin edges and number of bins
    xedges, muedges = edges
    nx = len(xedges) - 1
    nmu = len(muedges) - 1
    xdtype = max(xedges.dtype, muedges.dtype)
    # Always make sure first ell value is monopole, which is just (x, mu) projection since legendre of ell = 0 is 1
    return_poles = ells is not None
    ells = ells or []
    unique_ells = sorted(set([0]) | set(ells))
    legpoly = [legendre(ell) for ell in unique_ells]
    nell = len(unique_ells)

    # valid ell values
    if any(ell < 0 for ell in unique_ells):
        raise ValueError("Multipole numbers must be non-negative integers")

    # Initialize the binning arrays
    musum = np.zeros((nx + 3, nmu + 3))
    xsum = np.zeros((nx + 3, nmu + 3))
    ysum = np.zeros(
        (nell, nx + 3, nmu + 3), dtype=y3d.dtype
    )  # extra dimension for multipoles
    nsum = np.zeros((nx + 3, nmu + 3), dtype="i8")
    # If input array is Hermitian symmetric, only half of the last  axis is stored in `y3d`

    cellsize = (
        y3d.BoxSize / y3d.Nmesh
        if isinstance(y3d, RealField)
        else 2.0 * np.pi / y3d.BoxSize
    )
    mincell = np.min(cellsize)

    # Iterate over y-z planes of the coordinate mesh
    for islab in range(x3d[0].shape[0]):
        # The square of coordinate mesh norm
        # (either Fourier space k or configuraton space x)
        xvec = (x3d[0][islab].real.astype(xdtype),) + tuple(
            x3d[i].real.astype(xdtype) for i in range(1, 3)
        )
        xnorm = sum(xx ** 2 for xx in xvec) ** 0.5

        # If empty, do nothing
        if len(xnorm.flat) == 0:
            continue

        # Get the bin indices for x on the slab
        dig_x = np.digitize(xnorm.flat, xedges, right=False)
        mask_zero = xnorm < mincell / 2.0
        # y3d[islab, mask_zero[0]] = 0.
        dig_x[mask_zero.flat] = nx + 2

        # Get the bin indices for mu on the slab
        mu = sum(xx * ll for xx, ll in zip(xvec, los))
        mu[~mask_zero] /= xnorm[~mask_zero]

        if hermitian_symmetric == 0:
            mus = [mu]
        else:
            nonsingular = np.ones(xnorm.shape, dtype="?")
            # Get the indices that have positive freq along symmetry axis = -1
            nonsingular[...] = x3d[-1][0] > 0.0
            mus = [mu, -mu]

        # Accounting for negative frequencies
        for imu, mu in enumerate(mus):
            # Make the multi-index
            dig_mu = np.digitize(
                mu.flat, muedges, right=False
            )  # this is bins[i-1] <= x < bins[i]
            dig_mu[mu.real.flat == muedges[-1]] = nmu  # last mu inclusive
            dig_mu[mask_zero.flat] = nmu + 2

            multi_index = np.ravel_multi_index([dig_x, dig_mu], (nx + 3, nmu + 3))

            if hermitian_symmetric and imu:
                multi_index = multi_index[nonsingular.flat]
                xnorm = xnorm[nonsingular]  # it will be recomputed
                mu = mu[nonsingular]

            # Count number of modes in each bin
            nsum.flat += np.bincount(multi_index, minlength=nsum.size)
            # Sum up x in each bin
            xsum.flat += np.bincount(
                multi_index, weights=xnorm.flat, minlength=nsum.size
            )
            # Sum up mu in each bin
            musum.flat += np.bincount(multi_index, weights=mu.flat, minlength=nsum.size)

            # Compute multipoles by weighting by Legendre(ell, mu)
            for ill, ell in enumerate(unique_ells):

                weightedy3d = (2.0 * ell + 1.0) * legpoly[ill](mu)

                if hermitian_symmetric and imu:
                    # Weight the input 3D array by the appropriate Legendre polynomial
                    weightedy3d = (
                        hermitian_symmetric
                        * weightedy3d
                        * y3d[islab][nonsingular[0]].conj()
                    )  # hermitian_symmetric is 1 or -1
                else:
                    weightedy3d = weightedy3d * y3d[islab, ...]

                # Sum up the weighted y in each bin
                ysum[ill, ...].real.flat += np.bincount(
                    multi_index, weights=weightedy3d.real.flat, minlength=nsum.size
                )
                if np.iscomplexobj(ysum):
                    ysum[ill, ...].imag.flat += np.bincount(
                        multi_index, weights=weightedy3d.imag.flat, minlength=nsum.size
                    )

    # Sum binning arrays across all ranks
    xsum = comm.allreduce(xsum)
    musum = comm.allreduce(musum)
    ysum = comm.allreduce(ysum)
    nsum = comm.allreduce(nsum)

    # It is not clear how to proceed with beyond Nyquist frequencies
    # At Nyquist, kN = - pi * N / L (appears once in y3d.x) is the same as pi * N / L, so corresponds to mu and -mu
    # Our treatment of hermitian symmetric field would sum this frequency twice (mu and -mu)
    # But this would appear only once in uncompressed field
    # As a default, set frequencies beyond to NaN
    xmax = y3d.Nmesh // 2 * cellsize
    mask_beyond_nyq = np.flatnonzero(xedges >= np.min(xmax))
    xsum[mask_beyond_nyq] = np.nan
    musum[mask_beyond_nyq] = np.nan
    ysum[:, mask_beyond_nyq] = np.nan
    nsum[mask_beyond_nyq] = 0

    # Reshape and slice to remove out of bounds points
    sl = slice(1, -2)
    if not exclude_zero:
        dig_zero = tuple(
            np.digitize(0.0, edges, right=False) for edges in [xedges, muedges]
        )
        xsum[dig_zero] += xsum[nx + 2, nmu + 2]
        musum[dig_zero] += musum[nx + 2, nmu + 2]
        ysum[(Ellipsis,) + dig_zero] += ysum[:, nx + 2, nmu + 2]
        nsum[dig_zero] += nsum[nx + 2, nmu + 2]

    with np.errstate(invalid="ignore", divide="ignore"):

        # 2D binned results
        y2d = (ysum[0, ...] / nsum)[sl, sl]  # ell=0 is first index
        xmean2d = (xsum / nsum)[sl, sl]
        mumean2d = (musum / nsum)[sl, sl]
        n2d = nsum[sl, sl]
        zero2d = ysum[0, nx + 2, nmu + 2]

        # 1D multipole results (summing over mu (last) axis)
        if return_poles:
            n1d = nsum[sl, sl].sum(axis=-1)
            xmean1d = xsum[sl, sl].sum(axis=-1) / n1d
            poles = ysum[:, sl, sl].sum(axis=-1) / n1d
            poles_zero = ysum[:, nx + 2, nmu + 2]
            poles, poles_zero = (
                tmp[[unique_ells.index(ell) for ell in ells], ...]
                for tmp in (poles, poles_zero)
            )

    # Return y(x,mu) + (possibly empty) multipoles
    toret = [(xmean2d, mumean2d, y2d, n2d, zero2d)]
    toret.append((xmean1d, poles, n1d, poles_zero) if return_poles else None)
    return toret
