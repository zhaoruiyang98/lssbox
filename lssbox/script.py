from __future__ import annotations
import dataclasses
import logging
import mpi4py.MPI as MPI
import numpy as np
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Any, cast, Literal
from numpy import ndarray as NDArray
from nbodykit import CurrentMPIComm
from nbodykit.transform import StackColumns
from nbodykit.source.catalog.uniform import UniformCatalog
from nbodykit.source.catalog.file import FileCatalogBase
from nbodykit.source.catalog.file import CSVCatalog
from lssbox.pk import FFTPowerAP
from lssbox.recon import DisplacementSolver
from lssbox.recon import SafeFFTRecon


def _default_CSVkwargs():
    return {
        "names": ["xz", "yz", "zz", "x", "y", "z", "vx", "vy", "vz"],
        "dtype": "f8",
        "usecols": ["xz", "yz", "zz"],
        "delim_whitespace": True,
        "skiprows": 1,
    }


class Cloneable:
    def clone(self, **kwargs):
        dct = deepcopy(dataclasses.asdict(self))
        dct.update(kwargs)
        return type(self)(**dct)  # type: ignore


@dataclass
class DataReader(Cloneable):
    file: str = 'data/molino.z0.0.s8_p.nbody225.hod2_zrsd.ascii'
    reader: type[FileCatalogBase] = CSVCatalog
    position: list = field(default_factory=lambda: ['xz', 'yz', 'zz'])
    kwargs: dict = field(default_factory=_default_CSVkwargs)

    def load(self) -> FileCatalogBase:
        data = self.reader(self.file, **self.kwargs)
        data['Position'] = StackColumns(*[data[key] for key in self.position])
        return data


@dataclass
class ReconConfig(Cloneable):
    BoxSize: float | list[float] = 1000
    Nmesh: int = 512
    bias: float = 2.4
    f: float = 0.53
    los: list[int] = field(default_factory=lambda: [0, 0, 1])
    R: float = 10
    revert_rsd_random: bool = True
    scheme: str = 'LGS'
    dis_resampler: str = 'cic'
    dis_interlaced: bool = True
    dis_compensated: bool = False
    delta_resampler: str = 'cic'
    delta_interlaced: bool = True
    delta_compensated: bool = True

    def solve_displacement(
        self, data, ran, alperp: float = 1, alpara: float = 1,
    ) -> tuple[NDArray, NDArray]:
        solver = DisplacementSolver(
            dataA=data, ranA=ran, Nmesh=self.Nmesh, biasA=self.bias, f=self.f,
            los=self.los, R=self.R, revert_rsd_random=self.revert_rsd_random,
            BoxSize=self.BoxSize, resampler=self.dis_resampler,
            interlaced=self.dis_interlaced, compensated=self.dis_compensated,
        )
        solver.run(alperp=1, alpara=1)
        return solver.dis['dataA'], solver.dis['ranA']

    def recon(self, data, ran, s_d=None, s_r=None) -> SafeFFTRecon:
        if (s_d is None) or (s_r is None):
            s_d, s_r = self.solve_displacement(data, ran)
        recon = SafeFFTRecon(
            data=data, ran=ran, Nmesh=self.Nmesh, BoxSize=self.BoxSize,
            s_d=s_d, s_r=s_r, bias=self.bias, f=self.f, los=self.los, R=self.R,
            revert_rsd_random=self.revert_rsd_random,
            resampler=self.delta_resampler, interlaced=self.delta_interlaced,
            compensated=self.delta_compensated,
        )
        return recon


@dataclass
class APConfig(Cloneable):
    alperp: float = 1
    alpara: float = 1
    method: Literal["active", "passive"] = "active"

    def hasAP(self) -> bool:
        return True if (self.alperp != 1 or self.alpara != 1) else False


@contextmanager
def remove_BoxSize(data):
    data_BoxSize = data.attrs.pop('BoxSize', None)
    try:
        yield data
    finally:
        if data_BoxSize is not None:
            data.attrs['BoxSize'] = data_BoxSize


def collect_poles(res, shotfactor: float = 1) -> NDArray:
    names = [key for key in res.poles.variables if key.startswith("power_")]
    names.sort()
    k = res.poles['k']
    shot = res.attrs['shotnoise']
    out = np.empty((len(names) + 1, len(k)), dtype=np.float64)
    out[0] = k
    for i, v in enumerate([res.poles[name].real for name in names]):
        if i == 0:
            v -= shot * shotfactor
        out[i + 1, :] = v
    return out


@dataclass(eq=False)
class MeasurePower:
    type: Literal["pre", "post"] = "pre"
    AP: APConfig = field(default_factory=APConfig)
    recon: ReconConfig | None = None
    alpha: int = 10
    seed: int | None = None
    mode: str = '2d'
    BoxSize: float | list[float] = 1000
    Nmesh: int = 512
    los: list[int] = field(default_factory=lambda: [0, 0, 1])
    Nmu: int = 100
    poles: list[int] = field(default_factory=lambda: [0, 2, 4])
    dk: float = 0.01
    kmin: float = 0
    kmax: float = 1
    resampler: str = 'cic'
    interlaced: bool = True
    compensated: bool = True
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("MeasurePower"), repr=False)
    comm: MPI.Intracomm = field(init=False, repr=False)

    @CurrentMPIComm.enable
    def __post_init__(self, comm=None):
        self.comm = comm
        if self.type == "post":
            if self.recon is None:
                raise ValueError("recon is required when type is 'post'")
            if self.los != self.recon.los:
                raise ValueError("los must be the same as recon.los")
            if self.Nmesh != self.recon.Nmesh:
                self.mpi_warning(
                    "Nmesh=%s when doing FFTPower, "
                    "while Nmesh=%s when doing reconstruction",
                    self.Nmesh, self.recon.Nmesh)
            if self.AP.method == "passive":
                if self.BoxSize != self.recon.BoxSize:
                    raise ValueError(
                        "BoxSize must be the same as recon.BoxSize")

    @property
    def power_kwargs(self):
        return dict(
            mode=self.mode, Nmesh=self.Nmesh, los=self.los, Nmu=self.Nmu,
            poles=self.poles, dk=self.dk, kmin=self.kmin, kmax=self.kmax,
            interlaced=self.interlaced, compensated=self.compensated,
            resampler=self.resampler,
        )

    def measure(self, data, recover=True, ran=None):
        boxsize = data.attrs.get('BoxSize', None)
        if boxsize and (boxsize != self.BoxSize):
            raise ValueError("data's BoxSize must be the same as self.BoxSize")
        if self.type == "pre":
            out = self.measure_pre(data, recover)
        else:
            out = self.measure_post(data, recover, ran=ran)
        return out

    def measure_pre(self, data, recover=True):
        power_kwargs = self.power_kwargs
        alpara, alperp = self.AP.alpara, self.AP.alperp
        rescale = np.array([alpara if x == 1 else alperp for x in self.los])
        if self.AP.method == "passive":
            res = FFTPowerAP(
                data, BoxSize=self.BoxSize, alpara=alpara, alperp=alperp,
                **power_kwargs)
        else:
            data['Position'] /= rescale
            with remove_BoxSize(data):
                res = FFTPowerAP(
                    data, BoxSize=list(self.BoxSize / rescale), **power_kwargs)
            if recover:
                data['Position'] *= rescale
        return res

    def measure_post(self, data, recover=True, ran=None):
        assert self.recon is not None
        alpara, alperp = self.AP.alpara, self.AP.alperp
        rescale = np.array([alpara if x == 1 else alperp for x in self.los])
        _ = np.atleast_1d(self.BoxSize)
        if len(_) == 1:
            nbar = data.csize / _[0]**3
        else:
            nbar = data.csize / np.prod(_)
        if ran is None:
            ran = UniformCatalog(
                nbar * self.alpha, BoxSize=self.BoxSize, seed=self.seed)
        ran = cast(Any, ran)

        if self.AP.method == "passive":
            # transform to AP space
            rescaled_BoxSize = list(self.BoxSize / rescale)
            data['Position'] /= rescale
            ran['Position'] /= rescale
            ran.attrs['BoxSize'] = rescaled_BoxSize
            self.recon = self.recon.clone(BoxSize=rescaled_BoxSize)
            with remove_BoxSize(data):
                s_d, s_r = self.recon.solve_displacement(data, ran)
            # transform back
            self.recon = self.recon.clone(BoxSize=self.BoxSize)
            data['Position'] *= rescale
            ran['Position'] *= rescale
            recon = self.recon.recon(
                data, ran, s_d=s_d * rescale, s_r=s_r * rescale)
            # measure power spectrum
            res = FFTPowerAP(
                recon, BoxSize=self.BoxSize, alperp=alperp, alpara=alpara,
                **self.power_kwargs)
            res.attrs['shotnoise'] = (
                1 + 1 / self.alpha) * 1 / nbar / np.prod(rescale)
            if recover:
                ran.attrs['BoxSize'] = self.BoxSize
            # TODO: debug
            # # solve displacement field
            # s_d, s_r = self.recon.solve_displacement(
            #     data, ran, alperp=alperp, alpara=alpara)
            # # get reconstructed field
            # recon = self.recon.recon(data, ran, s_d=s_d, s_r=s_r)
            # # measure power spectrum
            # res = FFTPowerAP(
            #     recon, alperp=alperp, alpara=alpara, **self.power_kwargs)
            # res.attrs['shotnoise'] = (
            #     1 + 1 / self.alpha) * 1 / nbar / np.prod(rescale)
        else:
            # transform to AP space
            rescaled_BoxSize = list(self.BoxSize / rescale)
            data['Position'] /= rescale
            ran['Position'] /= rescale
            ran.attrs['BoxSize'] = rescaled_BoxSize
            if self.recon.BoxSize != rescaled_BoxSize:
                self.mpi_info(
                    'rescaled recon.BoxSize from %s to %s',
                    self.recon.BoxSize, rescaled_BoxSize)
                self.recon = self.recon.clone(BoxSize=rescaled_BoxSize)
            with remove_BoxSize(data):
                recon = self.recon.recon(data, ran)
                res = FFTPowerAP(
                    recon, BoxSize=rescaled_BoxSize, **self.power_kwargs)
            if recover:
                data['Position'] *= rescale
                ran['Position'] *= rescale
                ran.attrs['BoxSize'] = self.BoxSize
            res.attrs['shotnoise'] = (
                1 + 1 / self.alpha) * 1 / nbar / np.prod(rescale)
        self.mpi_warning(
            "shotnoise was computed assuming all Value=Weight=Selection=1")
        return res

    def mpi_warning(self, msg, *args):
        if self.comm.rank == 0:
            self.logger.warning(msg, *args)

    def mpi_info(self, msg, *args):
        if self.comm.rank == 0:
            self.logger.info(msg, *args)
