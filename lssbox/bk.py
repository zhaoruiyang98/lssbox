from __future__ import annotations
import numpy as np
from numpy import newaxis


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
