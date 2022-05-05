from __future__ import annotations
import logging
import warnings
import numpy as np
from typing import (
    cast,
    List,
    Any,
)
from numpy import ndarray as NDArray
from nbodykit import _global_options
from nbodykit.algorithms.fftrecon import FFTRecon
from nbodykit.base.catalog import CatalogSource
from nbodykit.source.mesh.catalog import get_compensation
from pmesh import window
from pmesh.pm import ParticleMesh
from pmesh.pm import RealField

Indice = List[int]


def create_density_field(
    cat: CatalogSource,
    s,
    pm: ParticleMesh,
    position: str = 'Position',
    resampler: str = 'cic',
    interlaced: bool = False,
    compensated: bool = False,
) -> RealField:
    delta = cast(RealField, pm.create(type='real', value=0))

    # ensure the slices are synced, since decomposition is collective
    Nlocalmax = max(pm.comm.allgather(cat.size))

    # python 2.7 wants floats.
    nbar = (1.0 * cat.csize / pm.Nmesh.prod())

    chunksize = _global_options['paint_chunk_size']

    for i in range(0, Nlocalmax, chunksize):
        sl = slice(i, i + chunksize)

        if s is not None:
            dpos = (
                cat[position].astype('f4')[sl] - s[sl]).compute()  # type: ignore
        else:
            dpos = (cat[position].astype('f4')[sl]).compute()  # type: ignore

        if not interlaced:
            if resampler == 'cic':
                # raw code
                # default behaviour (CIC)
                layout = pm.decompose(dpos)
                pm.paint(dpos, layout=layout, out=delta, hold=True)
            else:
                # support another resampler
                layout = pm.decompose(
                    dpos, smoothing=0.5 * window.methods[resampler].support)
                pm.paint(
                    dpos, layout=layout, out=delta, hold=True,
                    resampler=resampler)
        else:
            # for interlacing, we need two empty meshes if out was provided
            # since out may have non-zero elements, messing up our interlacing sum
            real1 = RealField(pm)
            real1[:] = 0
            # the second, shifted mesh (always needed)
            real2 = RealField(pm)
            real2[:] = 0

            if resampler == 'cic':
                # default behaviour (CIC)
                layout = pm.decompose(
                    dpos, smoothing=1.0 * window.methods['CIC'].support)
            else:
                # support another resampler
                layout = pm.decompose(
                    dpos, smoothing=1.0 * window.methods[resampler].support)

            # interlacing: use 2 meshes separated by 1/2 cell size
            # in mesh units
            shifted = pm.affine.shift(0.5)
            # paint to two shifted meshes
            pm.paint(dpos, resampler=resampler, hold=True, out=real1)
            pm.paint(dpos, resampler=resampler,
                     transform=shifted, hold=True, out=real2)
            # compose the two interlaced fields into the final result.
            c1 = real1.r2c()
            c2 = real2.r2c()
            H = pm.BoxSize / pm.Nmesh
            # and then combine
            for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                kH = sum(k[i] * H[i] for i in range(3))
                s1[...] = s1[...] * 0.5 + s2[...] * \
                    0.5 * np.exp(0.5 * 1j * kH)  # type: ignore
            # FFT back to real-space
            # NOTE: cannot use "delta" here in case user supplied "out"
            c1.c2r(real1)

            # need to add to the returned mesh if user supplied "out"
            delta[:] += real1[:]

    delta[...] /= nbar
    if compensated:
        action = get_compensation(interlaced, resampler)[0]
        kwargs = {}
        kwargs['func'] = action[1]
        if action[2] is not None:
            kwargs['kind'] = action[2]
        kwargs['out'] = Ellipsis
        delta = delta.r2c(out=Ellipsis)
        delta.apply(**kwargs)
        delta = delta.c2r(out=Ellipsis)

    return delta


class SafeFFTRecon(FFTRecon):
    r"""
    FFT based Lagrangian reconstruction algorithm in a periodic box.

    References:

        Eisenstein et al, 2007
        http://adsabs.harvard.edu/abs/2007ApJ...664..675E
        Section 3, paragraph starting with 'Restoring in full the ...'

    We follow a cleaner description in Schmitfull et al 2015,

        http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1508.06972

    Table I, and text below. Schemes are LGS, LF2 and LRR.

    A slight difference against the paper is that Redshift distortion
    and bias are corrected in the linear order. The Random shifting
    followed Martin White's suggestion to exclude the RSD by default.
    (with default `revert_rsd_random=False`.)

    Parameters
    ----------
    data : CatalogSource,
        the data catalog, e.g. halos. `data.attrs['BoxSize']` is used if argument `BoxSize` is not given.
    ran  :  CatalogSource
        the random catalog, e.g. from a `UniformCatalog` object.
    Nmesh : int
        The size of the FFT Mesh. Rule of thumb is that the size of a mesh cell
        shall be 2 ~ 4 times smaller than the smoothing length, `R`.
    data_indices: list[int], optional
        specify it if you want the displaced density field :math:`\delta_d` built from part of data particles, 
        while the displacement field is still estimated from the whole catalog, by default None
    ran_indices: list[int], optional
        specify it if you want the shifted density field :math:`\delta_s` built from part of random particles,
        while the displacement field is still estimated from the whole catalog, by default None
    s_d: ndarray, optional
        using precomputed s_d and s_r, by default None
    s_r: ndarray, optional
        using precomputed s_d and s_r, by default None
    bias : float
        The bias of the data catalog. by default 1.0
    f: float
        The growth rate; if non-zero, correct for RSD, by default 0.0
    los : list 
        The direction of the line of sight for RSD. Usually (default) [0, 0, 1].
    R : float
        The radius of smoothing. 10 to 20 Mpc/h is usually cool. by default 20
    position: string
        column to use for picking up the Position of the objects. by default 'Position'
    revert_rsd_random : boolean
        Revert the rsd for randoms as well as data. There are two conventions.
        either reverting rsd displacement in data displacement only(False) or
        in both data and randoms (True). Default is False.
    scheme : string
        The reconstruction scheme.
        `LGS` is the standard reconstruction (Lagrangian growth shift).
        `LF2` is the F2 Lagrangian reconstruction.
        `LRR` is the random-random Lagrangian reconstruction.
    BoxSize : float or array_like, optional
        the size of the periodic box, default is to infer from the data.
    resampler: str
        the string specifying which window interpolation scheme to use;
        see ``pmesh.window.methods``
        by default 'cic'
    interlaced: bool
        use the interlacing technique of Sefusatti et al. 2015 to reduce
        the effects of aliasing on Fourier space quantities computed from the mesh.
        by default False
    compensated: bool
        whether to correct for the window introduced by the grid
        interpolation scheme. by default False
    """

    def __init__(
        self,
        data: CatalogSource,
        ran: CatalogSource,
        Nmesh: int,
        data_indices: Indice | None = None,
        ran_indices: Indice | None = None,
        s_d: NDArray | None = None,
        s_r: NDArray | None = None,
        bias: float = 1.0,
        f: float = 0.0,
        los: list[int] = [0, 0, 1],
        R: float = 20,
        position: str = 'Position',
        revert_rsd_random: bool = False,
        scheme: str = 'LGS',
        BoxSize: float | list[float] | None = None,
        resampler: 'str' = 'cic',
        interlaced: bool = False,
        compensated: bool = False,
    ):
        super().__init__(
            data=data, ran=ran, Nmesh=Nmesh, bias=bias, f=f, los=los, R=R,
            position=position, revert_rsd_random=revert_rsd_random,
            scheme=scheme, BoxSize=BoxSize
        )
        self.attrs['interlaced'] = interlaced
        self.attrs['compensated'] = compensated
        self.attrs['resampler'] = str(resampler)
        self.data_indices: Indice | None = data_indices
        self.ran_indices: Indice | None = ran_indices
        self.s_d = s_d
        self.s_r = s_r
        self.displace: dict[str, Any] = {}

# dynamic attributes, copied from CatalogMesh
# =============================================================================>
    @property
    def interlaced(self) -> bool:
        return self.attrs['interlaced']

    @interlaced.setter
    def interlaced(self, interlaced: bool):
        self.attrs['interlaced'] = interlaced

    @property
    def window(self) -> str:
        return self.attrs['resampler']

    @window.setter
    def window(self, value: str):
        self.resampler = value

    @property
    def resampler(self) -> str:
        return self.attrs['resampler']

    @resampler.setter
    def resampler(self, value: str):
        assert value in window.methods
        self.attrs['resampler'] = value.lower()

    @property
    def compensated(self) -> bool:
        return self.attrs['compensated']

    @compensated.setter
    def compensated(self, value: bool):
        self.attrs['compensated'] = value
# <=============================================================================

    def run(self):
        if self.s_d is not None and self.s_r is not None:
            if self.comm.rank == 0:
                self.logger.info('using given displacement s_d and s_r')
            s_d, s_r = self.s_d, self.s_r
        else:
            s_d, s_r = self._compute_s()
        return self._helper_paint(s_d, s_r)

    def work_with(self, cat, s):
        return create_density_field(
            cat, s, self.pm,
            position=self.position, resampler=self.resampler,
            interlaced=self.interlaced, compensated=self.compensated
        )

    def _helper_paint(self, s_d, s_r):
        """ Convert the displacements of data and random to a single reconstruction mesh object. """

        def LGS(delta_s_r):
            if self.data_indices is None:
                delta_s_d = self.work_with(self.data, s_d)
            else:
                delta_s_d = self.work_with(
                    self.data[self.data_indices], s_d[self.data_indices])
            self._summary_field(delta_s_d, "delta_s_d (shifted)")

            delta_s_d[...] -= delta_s_r
            return delta_s_d

        def LRR(delta_s_r):
            delta_s_nr = self.work_with(self.ran, -s_r)
            self._summary_field(delta_s_nr, "delta_s_nr (reverse shifted)")

            delta_d = self.work_with(self.data, None)
            self._summary_field(delta_d, "delta_d (unshifted)")

            delta_s_nr[...] += delta_s_r[...]
            delta_s_nr[...] *= 0.5
            delta_d[...] -= delta_s_nr
            return delta_d

        def LF2(delta_s_r):
            lgs = LGS(delta_s_r)
            lrr = LRR(delta_s_r)
            lgs[...] *= 3.0 / 7.0
            lrr[...] *= 4.0 / 7.0
            lgs[...] += lrr
            return lgs

        if self.ran_indices is None:
            delta_s_r = self.work_with(self.ran, s_r)
        else:
            delta_s_r = self.work_with(
                self.ran[self.ran_indices], s_r[self.ran_indices])
        self._summary_field(delta_s_r, "delta_s_r (shifted)")

        if self.attrs['scheme'] == 'LGS':
            delta_recon = LGS(delta_s_r)
        elif self.attrs['scheme'] == 'LF2':
            delta_recon = LF2(delta_s_r)
        elif self.attrs['scheme'] == 'LRR':
            delta_recon = LRR(delta_s_r)
        else:
            raise ValueError(f"unsupported scheme = {self.attrs['scheme']}")

        self._summary_field(delta_recon, "delta_recon")

        # FIXME: perhaps change to 1 + delta for consistency. But it means loss of precision in f4
        return delta_recon


class DisplacementSolver:
    r"""compute the Zeldovich displacement

    if both dataB and ranB are given, multi-tracer displacements are computed.

    Notes
    -----
    results are stored in dictionary `self.dis`
    """

    def __init__(
        self,
        dataA: CatalogSource,
        ranA: CatalogSource,
        Nmesh: int,
        dataB: CatalogSource | None = None,
        ranB: CatalogSource | None = None,
        biasA: float = 1.0,
        biasB: float = 1.0,
        f: float = 0.0,
        los: list[int] = [0, 0, 1],
        R: float = 20,
        position: str = 'Position',
        revert_rsd_random: bool = False,
        BoxSize: float | list[float] | None = None,
        resampler: str = 'cic',
        interlaced: bool = False,
        compensated: bool = False,
    ) -> None:
        assert isinstance(dataA, CatalogSource)
        assert isinstance(ranA, CatalogSource)

        _los: NDArray = np.array(los, dtype='f8', copy=True)
        _los /= (_los ** 2).sum()
        assert len(_los) == 3
        assert (~np.isnan(_los)).all()

        assert position in dataA.columns
        assert position in ranA.columns
        self.position = position

        comm = dataA.comm
        self.comm = comm
        assert dataA.comm == ranA.comm
        self.mt = False
        if dataB:
            self.mt = True
        if self.mt:
            assert isinstance(dataB, CatalogSource)
            assert isinstance(ranB, CatalogSource)
            assert comm == dataB.comm == ranB.comm

        if Nmesh is None:
            Nmesh = dataA.attrs['Nmesh']
        _Nmesh = np.empty(3, dtype='i8')
        _Nmesh[...] = Nmesh

        if BoxSize is None:
            BoxSize = dataA.attrs['BoxSize']

        pmA = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh, comm=comm)
        pmB = None
        if self.mt:
            pmB = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh, comm=comm)
        self.pmA = pmA
        self.pmB = pmB

        if (self.pmA.BoxSize / self.pmA.Nmesh).max() > R:
            if comm.rank == 0:
                warnings.warn(
                    "The smoothing radius smaller than the mesh cell size. "
                    "This may produce undesired numerical results."
                )

        self.biasA = biasA
        self.biasB = biasB
        self.f = f
        self.los = _los
        self.R = R
        self.revert_rsd_random = bool(revert_rsd_random)
        self.resampler = resampler.lower()
        self.interlaced = interlaced
        self.compensated = compensated

        self.dataA = dataA
        self.ranA = ranA
        self.dataB = dataB
        self.ranB = ranB

        self.dis: dict[str, NDArray] = {}

        self.logger = logging.getLogger('DisplacementSolver')
        if self.comm.rank == 0:
            if not self.mt:
                self.logger.info(
                    "Solving displacement for bias=%g, f=%g, "
                    "smoothing R=%g los=%s",
                    self.biasA, self.f, self.R, self.los,
                )
            else:
                self.logger.info(
                    "Solving multi-tracer displacement for biasA=%g, biasB=%g, "
                    "f=%g, smoothing R=%g los=%s",
                    self.biasA, self.biasB, self.f, self.R, self.los,
                )

    def interpolate_displacement(self, cat, delta_d, pm, kernel):
        dpos = cat[self.position].astype('f4').compute()
        layout = pm.decompose(dpos)
        s_d = np.zeros_like(dpos, dtype='f4')

        for d in range(3):
            delta_d.apply(kernel(d)).c2r(out=...)\
                .readout(dpos, layout=layout, out=s_d[..., d])
        return s_d

    def create_delta_matter(self):
        def kernel(bias):
            def kernel(k, v):
                k2 = sum(ki**2 for ki in k)
                k2[k2 == 0] = 1.0  # type: ignore
                # reverting rsd.
                mu = sum(k[i] * self.los[i] for i in range(len(k))) / k2 ** 0.5
                frac = bias * (1 + self.f / bias * mu ** 2)
                return v / frac
            return kernel

        delta_d_A = create_density_field(
            self.dataA, None, self.pmA,
            position=self.position, resampler=self.resampler,
            interlaced=self.interlaced, compensated=self.compensated
        )
        delta_d_A = delta_d_A.r2c(out=Ellipsis).apply(kernel(self.biasA))
        if self.mt:
            assert self.dataB
            assert self.pmB
            delta_d_B = create_density_field(
                self.dataB, None, self.pmB,
                position=self.position, resampler=self.resampler,
                interlaced=self.interlaced, compensated=self.compensated
            )
            delta_d_B = delta_d_B.r2c(out=Ellipsis).apply(kernel(self.biasB))
            na = self.dataA.csize
            nb = self.dataB.csize
            delta_d_A[...] *= na / (na + nb)
            delta_d_A[...] += delta_d_B[...] * nb / (na + nb)
        return delta_d_A

    def run(self):
        def kernel(d):
            def kernel(k, v):
                k2 = sum(ki**2 for ki in k)
                k2[k2 == 0] = 1.0  # type: ignore
                v = v * np.exp(-0.5 * k2 * self.R**2)
                return 1j * k[d] / k2 * v
            return kernel

        delta_m = self.create_delta_matter()

        s_d_A = self.interpolate_displacement(
            self.dataA, delta_m,
            pm=self.pmA, kernel=kernel,
        )
        s_d_A_std = (self.comm.allreduce(
            (s_d_A**2).sum(axis=0)) / self.dataA.csize) ** 0.5
        if self.comm.rank == 0:
            self.logger.info(
                "Solved displacements of data%s, std(s_d) = %s",
                'A' if self.mt else '',
                str(s_d_A_std),
            )

        s_r_A = self.interpolate_displacement(
            self.ranA, delta_m,
            pm=self.pmA, kernel=kernel,
        )
        s_r_A_std = (self.comm.allreduce(
            (s_r_A**2).sum(axis=0)) / self.ranA.csize) ** 0.5
        if self.comm.rank == 0:
            self.logger.info(
                "Solved displacements of random%s, std(s_d) = %s",
                'A' if self.mt else '',
                str(s_r_A_std),
            )

        # convention 1
        s_d_A[...] *= (1 + self.los * self.f)
        # convention 2
        if self.revert_rsd_random:
            s_r_A[...] *= (1 + self.los * self.f)
        self.dis['dataA'] = s_d_A
        self.dis['ranA'] = s_r_A

        if self.mt:
            assert self.dataB
            assert self.ranB
            s_d_B = self.interpolate_displacement(
                self.dataB, delta_m,
                pm=self.pmB, kernel=kernel,
            )
            s_d_B_std = (self.comm.allreduce(
                (s_d_B**2).sum(axis=0)) / self.dataB.csize) ** 0.5
            if self.comm.rank == 0:
                self.logger.info(
                    "Solved displacements of data%s, std(s_d) = %s",
                    'B' if self.mt else '',
                    str(s_d_B_std),
                )

            s_r_B = self.interpolate_displacement(
                self.ranB, delta_m,
                pm=self.pmB, kernel=kernel,
            )
            s_r_B_std = (self.comm.allreduce(
                (s_r_B**2).sum(axis=0)) / self.ranB.csize) ** 0.5
            if self.comm.rank == 0:
                self.logger.info(
                    "Solved displacements of random%s, std(s_d) = %s",
                    'B' if self.mt else '',
                    str(s_r_B_std),
                )

            # convention 1
            s_d_B[...] *= (1 + self.los * self.f)
            # convention 2
            if self.revert_rsd_random:
                s_r_B[...] *= (1 + self.los * self.f)
            self.dis['dataB'] = s_d_B
            self.dis['ranB'] = s_r_B


if __name__ == '__main__':
    from nbodykit.lab import *  # type: ignore
    from nbodykit import setup_logging
    from nbodykit.binned_statistic import BinnedStatistic
    from nbodykit.source.catalog import CSVCatalog
    from nbodykit.transform import StackColumns

    setup_logging()

    alpha = 1
    dtype = 'f8'
    seed = 42
    boxsize = 1000
    Nmesh = 512
    bias = 2.4
    f = 0.53
    los = [0, 0, 1]
    R = 10

    names = ['xz', 'yz', 'zz', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    usecols = ['xz', 'yz', 'zz']
    data = CSVCatalog(
        'data/molino.z0.0.s8_p.nbody225.hod2_zrsd.ascii', names,
        dtype=dtype, usecols=usecols, delim_whitespace=True, skiprows=1,
    )
    data['Position'] = StackColumns(data['xz'], data['yz'], data['zz'])
    nbar = data.csize / boxsize**3

    ran = UniformCatalog(nbar * alpha, boxsize, seed=seed, dtype=dtype)

    solver = DisplacementSolver(
        dataA=data, ranA=ran, Nmesh=Nmesh,
        dataB=data, ranB=ran,
        biasA=bias, biasB=bias, f=f,
        los=los, R=R, BoxSize=boxsize, revert_rsd_random=True,
        resampler='cic', interlaced=True, compensated=True,
    )
    solver.run()

    reconA = SafeFFTRecon(
        data=data, ran=ran, Nmesh=Nmesh,
        s_d=solver.dis['dataA'], s_r=solver.dis['ranA'],
        bias=bias, f=f, los=los, R=R,
        revert_rsd_random=True, scheme='LGS', BoxSize=boxsize,
        resampler='cic', interlaced=True, compensated=True,
    )

    reconB = SafeFFTRecon(
        data=data, ran=ran, Nmesh=Nmesh,
        s_d=solver.dis['dataB'], s_r=solver.dis['ranB'],
        bias=bias, f=f, los=los, R=R,
        revert_rsd_random=True, scheme='LGS', BoxSize=boxsize,
        resampler='cic', interlaced=True, compensated=True,
    )

    res = FFTPower(
        reconA, second=reconB,
        mode='2d', Nmesh=512, BoxSize=boxsize,
        los=los, Nmu=100, poles=[0, 2, 4], dk=0.01,
    )

    poles = cast(BinnedStatistic, res.poles)
    arr = np.vstack(
        (poles['k'], poles['power_0'].real,
         poles['power_2'].real, poles['power_4'].real)
    ).T
    np.savetxt(
        f'res/reconAxB_ran{alpha}xdata_Nmesh{Nmesh}_S{R}_zrsd_cic_comT_inlT.txt',
        arr,
        # header=((5 * ' ').join(('k', 'P0', 'P2', 'P4'))
        #         + f'\nPshot={1./(data.csize/boxsize**3)}')
        header=((5 * ' ').join(('k', 'P0', 'P2', 'P4')))
    )
