from __future__ import annotations
import numpy as np
from contextlib import contextmanager
from mpi4py import MPI
from numpy import ndarray as NDArray
from numpy import newaxis
from scipy.interpolate import Rbf
from scipy.special import legendre


def AP(
    lBk: list[NDArray],
    k1: NDArray,
    k2: NDArray,
    k3: NDArray,
    mu1: NDArray,
    phi: NDArray,
    alperp: float,
    alpara: float,
):
    """compute approximate AP effect of B0 using m=0 modes only
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    if rank == 0:
        F = alpara / alperp

        # Define mu1, mu2, mu3 angles
        eta12 = (k3 ** 2.0 - k1 ** 2.0 - k2 ** 2.0) / (2.0 * k1 * k2)
        eta12_ = np.sqrt(np.abs(1.0 - eta12 ** 2.0))

        # k, n, m : k, mu1, phi
        mu2 = eta12[:, newaxis, newaxis] * mu1[newaxis, :, newaxis] - np.sqrt(
            1.0 - mu1[newaxis, :, newaxis] ** 2.0
        ) * eta12_[:, newaxis, newaxis] * np.cos(phi[newaxis, newaxis, :])
        mu3 = (
            -(k2 / k3)[:, newaxis, newaxis] * mu2[:, :, :]
            - (k1 / k3)[:, newaxis, newaxis] * mu1[newaxis, :, newaxis]
        )

        def qnu(k, mu) -> tuple[NDArray, NDArray]:
            q = k / alperp * (1 + mu ** 2 * (F ** (-2) - 1)) ** 0.5
            nu = mu / F * (1 + mu ** 2 * (F ** (-2) - 1)) ** (-0.5)
            return q, nu

        q1, nu1 = qnu(k1[:, newaxis, newaxis], mu1[newaxis, :, newaxis])
        q2, _ = qnu(k2[:, newaxis, newaxis], mu2[:, :, :])
        q3, _ = qnu(k3[:, newaxis, newaxis], mu3[:, :, :])
        tmp = np.ones(q2.shape)
        q1 = q1[:, :, :] * tmp
        nu1 = nu1[:, :, :] * tmp

        def split(arr):
            return np.array_split(arr, nranks, axis=0)

        q1, q2, q3, nu1 = (split(x) for x in (q1, q2, q3, nu1))

    else:
        q1, q2, q3, nu1 = None, None, None, None

    q1, q2, q3, nu1 = (comm.scatter(x, root=0) for x in (q1, q2, q3, nu1))

    nl = len(lBk)
    ells = [2 * i for i in range(nl)]
    Bkfns = [Rbf(k1, k2, k3, Blk, function="cubic") for Blk in lBk]
    Bks = []
    for fn in Bkfns:
        ni, nj, nk = q1.shape
        out = np.zeros(q1.shape)
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    out[i, j, k] = float(fn(q1[i, j, k], q2[i, j, k], q3[i, j, k]))

        Bks.append(out)
    # Bks = [fn(q1, q2, q3) for fn in Bksfns] # memory issue
    legends = [legendre(ell)(nu1) for ell in ells]
    Bk = sum(x * y for x, y in zip(Bks, legends))

    Bk = comm.gather(Bk, root=0)
    if rank == 0:
        Bk = np.vstack(Bk)
    Bk = comm.bcast(Bk, root=0)

    out = np.trapz(Bk, phi, axis=-1)
    out = np.trapz(out, mu1, axis=-1)
    out = out / (alpara ** 2 * alperp ** 4 * 4 * np.pi)
    return out


@contextmanager
def main_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        yield
    else:
        yield


@contextmanager
def timer(name: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        start = MPI.Wtime()
        yield
        end = MPI.Wtime()
        print(f"{name}: {end - start}")
    else:
        yield


def rebinning(
    file: str, kstart: float, kend: float, kbin_out: float, atol: float = 1e-6
):
    """
    bispectrum monopole rebinning

    Parameters
    ----------
    file : str
        path to data file
        should contain columns: k1, k2, k3, Bk, Nmode
        k1 <= k2 <= k3
    kstart : float
        start of k edges
    kend : float
        end of k edges
    kbin_out : float
        output k bin width
    atol : float
        tolerance for floating point comparison
    
    Returns
    -------
    np.ndarray
    """
    data = np.loadtxt(file)
    ncols = data.shape[-1]
    kedges = np.arange(kstart, kend, kbin_out)  # start
    nbins = kedges.size
    discard = False
    if kedges[-1] + kbin_out - kend > atol:
        print("discard the last bin, since raw data does not cover it")
        discard = True

    nrows = 0
    ii = 0
    lookup: dict[tuple[int, int, int], int] = {}
    for i in range(1, nbins + 1):
        for j in range(1, i + 1):
            for k in list(range(max(i - j, 1), j + 1))[::-1]:
                lookup[(i, j, k)] = ii
                nrows += 1
                ii += 1

    out = np.zeros((nrows, ncols), dtype=np.float64)
    modes = np.zeros(nrows, dtype=int)

    ijk = np.searchsorted(kedges, data[:, [0, 1, 2]])
    ijk = [tuple(arr) for arr in ijk]
    nskip = 0
    for idx, arr in zip(ijk, data):
        ii = lookup.get(idx, None)
        if ii is None:
            nskip += 1
            continue  # discard configuration not appearing in lookup table
        weight = arr[-1]
        if discard:
            if nbins in idx:
                weight = 0
        out[ii] += arr * weight  # weight by Nmode
        modes[ii] += arr[-1]
    if nskip > 0:
        print(f"{nskip} configurations are discarded")
    zero_modes = modes[modes == 0].size
    if zero_modes > 0:
        print(f"{zero_modes} modes are zero, missing configurations in raw data file")
    out[...] /= modes[:, newaxis]
    out[:, -1] = modes
    out = out[out[:, 0] != 0]

    return out


if __name__ == "__main__":
    path = "data/bks0_kbin0.01_thbin1_cic_m13_run00020_ng1024_001_kmax0p4.txt"
    out = rebinning(path, 0.01, 0.40, 0.02)
    np.savetxt("data/test.txt", out)
