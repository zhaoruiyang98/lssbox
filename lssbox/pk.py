from __future__ import annotations
import numpy
from contextlib import contextmanager
from typing import cast
from nbodykit.algorithms.fftpower import _find_unique_edges
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.algorithms.fftpower import project_to_basis
from nbodykit.base.catalog import CatalogSourceBase
from nbodykit.base.mesh import MeshSource
from pmesh.pm import ComplexField


@contextmanager
def rescale_y3d(
    y3d: ComplexField, *,
    los=[0, 0, 1], alpara: float = 1.0, alperp: float = 1.0,
):
    rawx = [xx.copy() for xx in y3d.x]
    for xx, is_z in zip(y3d.x, los):
        xx[:] *= alpara if is_z else alperp
    yield y3d
    y3d.x = rawx


def _cast_source(
    source, BoxSize, Nmesh,
    resampler='cic', interlaced=False, compensated=True,
):
    from pmesh.pm import Field
    from nbodykit.source.mesh import FieldMesh

    if isinstance(source, Field):
        source = FieldMesh(source)
    elif isinstance(source, CatalogSourceBase):
        if not isinstance(source, MeshSource):
            source = source.to_mesh(
                BoxSize=BoxSize, Nmesh=Nmesh, dtype='f8', resampler=resampler,
                interlaced=interlaced, compensated=compensated)
    if not isinstance(source, MeshSource):
        raise TypeError(f'Unknow type of source in FFTPowerAP: {type(source)}')
    if BoxSize is not None and any(source.attrs['BoxSize'] != BoxSize):
        raise ValueError(
            'Mismatched BoxSize between __init__ and source.attrs')
    if Nmesh is not None and any(source.attrs['Nmesh'] != Nmesh):
        raise ValueError(
            'Mismatched Nmesh between __init__ and source.attrs;'
            'if trying to re-sample with a different mesh, specify '
            '`Nmesh` as keyword of to_mesh()')

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

    Notes
    -----
    shot noise and power spectrum has been rescaled by 1 / (alperp**2 * alpara)
    """

    def __init__(
        self, first, mode, Nmesh=None, BoxSize=None, second=None,
        los=[0, 0, 1], Nmu=5, dk=None, kmin=0., kmax=None, poles=[],
        alpara=1.0, alperp=1.0, resampler='cic',
        interlaced=False, compensated=True,
    ):
        self.alpara, self.alperp = alpara, alperp
        if not (los == [0, 0, 1] or los == [0, 1, 0] or los == [1, 0, 0]):
            raise ValueError(
                "LOS must be [0, 0, 1], [0, 1, 0] or [1, 0, 0].")

        # copied from FFTPower, check first!
        # mode is either '1d' or '2d'
        if mode not in ['1d', '2d']:
            raise ValueError("`mode` should be either '1d' or '2d'")

        # check los
        if numpy.isscalar(los) or len(los) != 3:
            raise ValueError(
                "line-of-sight ``los`` should be vector with length 3")
        if not numpy.allclose(numpy.einsum('i,i', los, los), 1.0, rtol=1e-5):
            raise ValueError("line-of-sight ``los`` must be a unit vector")
        # ----------------------------------
        # perform cast_source here
        first = _cast_source(
            first, Nmesh=Nmesh, BoxSize=BoxSize, resampler=resampler,
            interlaced=interlaced, compensated=compensated)
        if second is not None:
            second = _cast_source(
                second, Nmesh=Nmesh, BoxSize=BoxSize, resampler=resampler,
                interlaced=interlaced, compensated=compensated)
        else:
            second = first

        super().__init__(
            first, mode, Nmesh, BoxSize, second,
            los, Nmu, dk, kmin, kmax, poles,
        )
        self.attrs['shotnoise'] /= alperp ** 2 * alpara

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
        if self.attrs['mode'] == "1d":
            self.attrs['Nmu'] = 1

        # measure the 3D power (y3d is a ComplexField)
        y3d, attrs = self._compute_3d_power(self.first, self.second)
        y3d = cast(ComplexField, y3d)

        # binning in k out to the minimum nyquist frequency
        # (accounting for possibly anisotropic box)
        dk = self.attrs['dk']
        kmin = self.attrs['kmin']
        kmax = self.attrs['kmax']
        if kmax is None:
            kmax = numpy.pi * y3d.Nmesh.min() / y3d.BoxSize.max() + dk / 2

        if dk > 0:
            kedges = numpy.arange(kmin, kmax, dk)
            kcoords = None
        else:
            raise NotImplementedError
            kedges, kcoords = _find_unique_edges(
                y3d.x, 2 * numpy.pi / y3d.BoxSize, kmax, y3d.pm.comm)

        # project on to the desired basis
        muedges = numpy.linspace(-1, 1, self.attrs['Nmu'] + 1, endpoint=True)
        edges = [kedges, muedges]
        coords = [kcoords, None]
        with rescale_y3d(
            y3d, los=self.attrs['los'], alpara=self.alpara, alperp=self.alperp
        ) as y3d:
            result, pole_result = project_to_basis(y3d, edges,
                                                   poles=self.attrs['poles'],
                                                   los=self.attrs['los'])

        # format the power results into structured array
        if self.attrs['mode'] == "1d":
            cols = ['k', 'power', 'modes']
            icols = [0, 2, 3]
            edges = edges[0:1]
            coords = coords[0:1]
        else:
            cols = ['k', 'mu', 'power', 'modes']
            icols = [0, 1, 2, 3]

        # power results as a structured array
        dtype = numpy.dtype([(name, result[icol].dtype.str)
                            for icol, name in zip(icols, cols)])
        power = numpy.squeeze(numpy.empty(result[0].shape, dtype=dtype))
        for icol, col in zip(icols, cols):
            power[col][:] = numpy.squeeze(result[icol])
        power['power'] /= self.alperp**2 * self.alpara

        # multipole results as a structured array
        poles = None
        if pole_result is not None:
            k, poles, N = pole_result
            cols = ['k'] + ['power_%d' %
                            l for l in self.attrs['poles']] + ['modes']
            result = [k] + [pole for pole in poles] + [N]

            dtype = numpy.dtype([(name, result[icol].dtype.str)
                                for icol, name in enumerate(cols)])
            poles = numpy.empty(result[0].shape, dtype=dtype)
            for icol, col in enumerate(cols):
                poles[col][:] = result[icol]
            for key in [f'power_{l}' for l in self.attrs['poles']]:
                poles[key] /= self.alperp**2 * self.alpara

        return self._make_datasets(edges, poles, power, coords, attrs)
