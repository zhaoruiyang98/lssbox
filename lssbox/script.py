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
    file: str = "data/molino.z0.0.s8_p.nbody225.hod2_zrsd.ascii"
    reader: type[FileCatalogBase] = CSVCatalog
    position: list = field(default_factory=lambda: ["xz", "yz", "zz"])
    kwargs: dict = field(default_factory=_default_CSVkwargs)

    def load(self) -> FileCatalogBase:
        data = self.reader(self.file, **self.kwargs)
        data["Position"] = StackColumns(*[data[key] for key in self.position])
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
    scheme: str = "LGS"
    dis_resampler: str = "cic"
    dis_interlaced: bool = True
    dis_compensated: bool = False
    delta_resampler: str = "cic"
    delta_interlaced: bool = True
    delta_compensated: bool = True
    data_idx: list[int] | None = None
    ran_idx: list[int] | None = None

    def solve_displacement(
        self, data, ran, alperp: float = 1, alpara: float = 1,
    ) -> tuple[NDArray, NDArray]:
        solver = DisplacementSolver(
            dataA=data,
            ranA=ran,
            Nmesh=self.Nmesh,
            biasA=self.bias,
            f=self.f,
            los=self.los,
            R=self.R,
            revert_rsd_random=self.revert_rsd_random,
            BoxSize=self.BoxSize,
            resampler=self.dis_resampler,
            interlaced=self.dis_interlaced,
            compensated=self.dis_compensated,
        )
        solver.run(alperp=alperp, alpara=alpara)
        return solver.dis["dataA"], solver.dis["ranA"]

    def recon(self, data, ran, s_d=None, s_r=None) -> SafeFFTRecon:
        if (s_d is None) or (s_r is None):
            s_d, s_r = self.solve_displacement(data, ran)
        recon = SafeFFTRecon(
            data=data,
            ran=ran,
            Nmesh=self.Nmesh,
            BoxSize=self.BoxSize,
            s_d=s_d,
            s_r=s_r,
            bias=self.bias,
            f=self.f,
            los=self.los,
            R=self.R,
            revert_rsd_random=self.revert_rsd_random,
            resampler=self.delta_resampler,
            interlaced=self.delta_interlaced,
            compensated=self.delta_compensated,
            data_indices=self.data_idx,
            ran_indices=self.ran_idx,
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
    data_BoxSize = data.attrs.pop("BoxSize", None)
    try:
        yield data
    finally:
        if data_BoxSize is not None:
            data.attrs["BoxSize"] = data_BoxSize


def collect_poles(res, shotfactor: float = 1) -> NDArray:
    names = [key for key in res.poles.variables if key.startswith("power_")]
    names.sort()
    k = res.poles["k"]
    shot = res.attrs["shotnoise"]
    out = np.empty((len(names) + 1, len(k)), dtype=np.float64)
    out[0] = k
    for i, v in enumerate([res.poles[name].real for name in names]):
        if i == 0:
            v -= shot * shotfactor
        out[i + 1, :] = v
    return out


@dataclass
class SimulationPower:
    """Measure power spectrum with AP effect in a simulation box.
    """

    alperp: float = 1
    alpara: float = 1
    APmethod: Literal["active", "passive"] = "active"
    mode: str = "2d"
    BoxSize: float | list[float] = 1000
    Nmesh: int = 512
    los: list[int] = field(default_factory=lambda: [0, 0, 1])
    Nmu: int = 100
    poles: list[int] = field(default_factory=lambda: [0, 2, 4])
    dk: float = 0.01
    kmin: float = 0
    kmax: float = 1
    resampler: str = "cic"
    interlaced: bool = True
    compensated: bool = True
    arnold: bool = False
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("SimulationPower"),
        repr=False,
        compare=False,
    )
    comm: MPI.Intracomm = field(init=False, repr=False, compare=False)

    @CurrentMPIComm.enable
    def __post_init__(self, comm=None):
        self.comm = comm

    @property
    def power_kwargs(self):
        return dict(
            mode=self.mode,
            Nmesh=self.Nmesh,
            los=self.los,
            Nmu=self.Nmu,
            poles=self.poles,
            dk=self.dk,
            kmin=self.kmin,
            kmax=self.kmax,
            interlaced=self.interlaced,
            compensated=self.compensated,
            resampler=self.resampler,
            arnold=self.arnold,
        )

    @property
    def rescale(self):
        return np.array([self.alpara if x == 1 else self.alperp for x in self.los])

    def has_AP(self) -> bool:
        return True if (self.alperp != 1 or self.alpara != 1) else False

    def mpi_warning(self, msg, *args):
        if self.comm.rank == 0:
            self.logger.warning(msg, *args)

    def mpi_info(self, msg, *args):
        if self.comm.rank == 0:
            self.logger.info(msg, *args)

    def measure(self, data, recover=True):
        raise NotImplementedError


@dataclass
class PrePower(SimulationPower):
    """Measure pre-recon power spectrum with AP effect in a simulation box.
    """

    def __post_init__(self, comm=None):
        super().__post_init__(comm=comm)
        self.logger = logging.getLogger("PrePower")

    def measure(self, data, recover=True):
        boxsize = data.attrs.get("BoxSize", None)
        if boxsize and (boxsize != self.BoxSize):
            raise ValueError("data's BoxSize must be the same as self.BoxSize")

        power_kwargs = self.power_kwargs
        alpara, alperp, rescale = self.alpara, self.alperp, self.rescale
        if self.APmethod == "passive":
            res = FFTPowerAP(
                data, BoxSize=self.BoxSize, alpara=alpara, alperp=alperp, **power_kwargs
            )
        else:
            data["Position"] /= rescale
            with remove_BoxSize(data):
                res = FFTPowerAP(
                    data, BoxSize=list(self.BoxSize / rescale), **power_kwargs
                )
            if recover:
                data["Position"] *= rescale
        return res


@dataclass
class PostPower(SimulationPower):
    """Measure post-recon power spectrum with AP effect in a simulation box.
    """

    recon: ReconConfig = field(default_factory=ReconConfig)
    alpha: int = 10
    seed: int | None = None

    def __post_init__(self, comm=None):
        super().__post_init__(comm=comm)
        self.logger = logging.getLogger("PostPower")
        if self.los != self.recon.los:
            raise ValueError("los must be the same as recon.los")
        if self.Nmesh != self.recon.Nmesh:
            self.mpi_warning(
                "Nmesh=%s when doing FFTPower, "
                "while Nmesh=%s when doing reconstruction",
                self.Nmesh,
                self.recon.Nmesh,
            )
        if (self.APmethod == "passive") and (self.BoxSize != self.recon.BoxSize):
            raise ValueError("BoxSize must be the same as recon.BoxSize")

    def measure(self, data, recover=True, ran=None):
        boxsize = data.attrs.get("BoxSize", None)
        if boxsize and (boxsize != self.BoxSize):
            raise ValueError("data's BoxSize must be the same as self.BoxSize")

        boxsize = np.atleast_1d(self.BoxSize)
        if len(boxsize) == 1:
            nbar = data.csize / boxsize ** 3
        else:
            nbar = data.csize / np.prod(boxsize)
        if ran is None:
            ran = UniformCatalog(
                nbar * self.alpha, BoxSize=self.BoxSize, seed=self.seed
            )
        ran = cast(Any, ran)

        alperp, alpara, rescale = self.alperp, self.alpara, self.rescale
        if self.APmethod == "passive":
            # transform to AP space
            rescaled_BoxSize = list(self.BoxSize / rescale)
            data["Position"] /= rescale
            ran["Position"] /= rescale
            ran.attrs["BoxSize"] = rescaled_BoxSize
            self.recon = self.recon.clone(BoxSize=rescaled_BoxSize)
            with remove_BoxSize(data):
                s_d, s_r = self.recon.solve_displacement(data, ran)
            # transform back
            self.recon = self.recon.clone(BoxSize=self.BoxSize)
            data["Position"] *= rescale
            ran["Position"] *= rescale
            recon = self.recon.recon(data, ran, s_d=s_d * rescale, s_r=s_r * rescale)
            # measure power spectrum
            res = FFTPowerAP(
                recon,
                BoxSize=self.BoxSize,
                alperp=alperp,
                alpara=alpara,
                **self.power_kwargs,
            )
            res.attrs["shotnoise"] = (1 + 1 / self.alpha) * 1 / nbar / np.prod(rescale)
            if recover:
                ran.attrs["BoxSize"] = self.BoxSize
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
            data["Position"] /= rescale
            ran["Position"] /= rescale
            ran.attrs["BoxSize"] = rescaled_BoxSize
            if self.recon.BoxSize != rescaled_BoxSize:
                self.mpi_info(
                    "rescaled recon.BoxSize from %s to %s",
                    self.recon.BoxSize,
                    rescaled_BoxSize,
                )
                self.recon = self.recon.clone(BoxSize=rescaled_BoxSize)
            with remove_BoxSize(data):
                recon = self.recon.recon(data, ran)
                res = FFTPowerAP(recon, BoxSize=rescaled_BoxSize, **self.power_kwargs)
            if recover:
                data["Position"] *= rescale
                ran["Position"] *= rescale
                ran.attrs["BoxSize"] = self.BoxSize
            res.attrs["shotnoise"] = (1 + 1 / self.alpha) * 1 / nbar / np.prod(rescale)
        self.mpi_warning("shotnoise was computed assuming all Value=Weight=Selection=1")
        return res


def half_split(size: int, seed=None) -> tuple[NDArray, NDArray]:
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(size, size, replace=False)
    x1, x2 = idx[: size // 2], idx[size // 2 :]
    x1.sort(), x2.sort()
    return x1, x2


@dataclass
class CrossPower(SimulationPower):
    """Measure cross power spectrum with AP effect in a simulation box.
    """

    recon: ReconConfig = field(default_factory=ReconConfig)
    alpha: int = 10
    seed: int | None = None
    split: bool = False
    data_idx1: NDArray | None = None
    ran_idx1: NDArray | None = None
    data_idx2: NDArray | None = None
    ran_idx2: NDArray | None = None

    def __post_init__(self, comm=None):
        super().__post_init__(comm=comm)
        self.logger = logging.getLogger("CrossPower")
        if self.los != self.recon.los:
            raise ValueError("los must be the same as recon.los")
        if self.Nmesh != self.recon.Nmesh:
            self.mpi_warning(
                "Nmesh=%s when doing FFTPower, "
                "while Nmesh=%s when doing reconstruction",
                self.Nmesh,
                self.recon.Nmesh,
            )
        if (self.APmethod == "passive") and (self.BoxSize != self.recon.BoxSize):
            raise ValueError("BoxSize must be the same as recon.BoxSize")
        if self.split:
            self.mpi_info(
                "measuring cross power spectrum via "
                "half-sum half-difference approach"
            )

    def _update_random_indices(self, data, ran) -> None:
        given_indices = all(
            [
                x is not None
                for x in (self.data_idx1, self.data_idx2, self.ran_idx1, self.ran_idx2)
            ]
        )
        if self.split and not given_indices:
            self.mpi_info("split is True, but missing indices, regenerated")
            data_idx1, data_idx2 = half_split(data.csize, seed=self.seed)
            ran_idx1, ran_idx2 = half_split(ran.csize, seed=self.seed)
            self.data_idx1, self.data_idx2 = data_idx1, data_idx2
            self.ran_idx1, self.ran_idx2 = ran_idx1, ran_idx2

    def measure(self, data, recover=True, ran=None):
        boxsize = data.attrs.get("BoxSize", None)
        if boxsize and (boxsize != self.BoxSize):
            raise ValueError("data's BoxSize must be the same as self.BoxSize")

        boxsize = np.atleast_1d(self.BoxSize)
        if len(boxsize) == 1:
            nbar = data.csize / boxsize ** 3
        else:
            nbar = data.csize / np.prod(boxsize)
        if ran is None:
            ran = UniformCatalog(
                nbar * self.alpha, BoxSize=self.BoxSize, seed=self.seed
            )
        ran = cast(Any, ran)
        self._update_random_indices(data, ran)

        alperp, alpara, rescale = self.alperp, self.alpara, self.rescale
        if self.APmethod == "passive":
            # transform to AP space
            rescaled_BoxSize = list(self.BoxSize / rescale)
            data["Position"] /= rescale
            ran["Position"] /= rescale
            ran.attrs["BoxSize"] = rescaled_BoxSize
            self.recon = self.recon.clone(BoxSize=rescaled_BoxSize)
            with remove_BoxSize(data):
                s_d, s_r = self.recon.solve_displacement(data, ran)
            # transform back
            self.recon = self.recon.clone(BoxSize=self.BoxSize)
            data["Position"] *= rescale
            ran["Position"] *= rescale
            if not self.split:
                recon = self.recon.recon(
                    data, ran, s_d=s_d * rescale, s_r=s_r * rescale
                )
                # measure power spectrum
                res = FFTPowerAP(
                    recon,
                    second=data,
                    BoxSize=self.BoxSize,
                    alperp=alperp,
                    alpara=alpara,
                    **self.power_kwargs,
                )
            else:
                recon1 = self.recon.clone(
                    data_idx=self.data_idx1, ran_idx=self.ran_idx1
                ).recon(data, ran, s_d=s_d * rescale, s_r=s_r * rescale)
                recon2 = self.recon.clone(
                    data_idx=self.data_idx2, ran_idx=self.ran_idx2
                ).recon(data, ran, s_d=s_d * rescale, s_r=s_r * rescale)
                pre1 = data[self.data_idx1]
                pre2 = data[self.data_idx2]
                res12 = FFTPowerAP(
                    recon1,
                    second=pre2,
                    BoxSize=self.BoxSize,
                    alperp=alperp,
                    alpara=alpara,
                    **self.power_kwargs,
                )
                res21 = FFTPowerAP(
                    recon2,
                    second=pre1,
                    BoxSize=self.BoxSize,
                    alperp=alperp,
                    alpara=alpara,
                    **self.power_kwargs,
                )
                res = res12, res21
            if recover:
                ran.attrs["BoxSize"] = self.BoxSize
        else:
            # transform to AP space
            rescaled_BoxSize = list(self.BoxSize / rescale)
            data["Position"] /= rescale
            ran["Position"] /= rescale
            ran.attrs["BoxSize"] = rescaled_BoxSize
            if self.recon.BoxSize != rescaled_BoxSize:
                self.mpi_info(
                    "rescaled recon.BoxSize from %s to %s",
                    self.recon.BoxSize,
                    rescaled_BoxSize,
                )
                self.recon = self.recon.clone(BoxSize=rescaled_BoxSize)
            with remove_BoxSize(data):
                if not self.split:
                    recon = self.recon.recon(data, ran)
                    res = FFTPowerAP(
                        recon,
                        second=data,
                        BoxSize=rescaled_BoxSize,
                        **self.power_kwargs,
                    )
                else:
                    s_d, s_r = self.recon.solve_displacement(data, ran)
                    recon1 = self.recon.clone(
                        data_idx=self.data_idx1, ran_idx=self.ran_idx1
                    ).recon(data, ran, s_d=s_d, s_r=s_r)
                    recon2 = self.recon.clone(
                        data_idx=self.data_idx2, ran_idx=self.ran_idx2
                    ).recon(data, ran, s_d=s_d, s_r=s_r)
                    pre1 = data[self.data_idx1]
                    pre2 = data[self.data_idx2]
                    res12 = FFTPowerAP(
                        recon1,
                        second=pre2,
                        BoxSize=rescaled_BoxSize,
                        **self.power_kwargs,
                    )
                    res21 = FFTPowerAP(
                        recon2,
                        second=pre1,
                        BoxSize=rescaled_BoxSize,
                        **self.power_kwargs,
                    )
                    res = res12, res21
            if recover:
                data["Position"] *= rescale
                ran["Position"] *= rescale
                ran.attrs["BoxSize"] = self.BoxSize
        return res


if __name__ == "__main__":
    from nbodykit import setup_logging

    def compute_mean(res):
        out1 = collect_poles(res[0])
        out2 = collect_poles(res[1])
        out1[1:, :] += out2[1:, :]
        out1[1:, :] /= 2
        return out1

    setup_logging()
    reader = DataReader("data/molino.z0.0.s8_p.nbody225.hod2_zrsd.ascii")
    data = reader.load()

    seed = 42
    ran = UniformCatalog(data.csize / 1000 ** 3 * 10, BoxSize=1000, seed=seed)

    alperp, alpara = 1.1, 1.2
    active_cross = CrossPower(
        alpara=alpara, alperp=alperp, APmethod="active", split=True, seed=seed
    )
    pk_active = compute_mean(active_cross.measure(data, ran=ran))

    passive_cross = CrossPower(
        alpara=alpara,
        alperp=alperp,
        APmethod="passive",
        split=True,
        data_idx1=active_cross.data_idx1,
        data_idx2=active_cross.data_idx2,
        ran_idx1=active_cross.ran_idx1,
        ran_idx2=active_cross.ran_idx2,
        seed=seed,
    )
    pk_passive = compute_mean(passive_cross.measure(data, ran=ran))

    np.savetxt("cross_active.txt", pk_active)
    np.savetxt("cross_passive.txt", pk_passive)

