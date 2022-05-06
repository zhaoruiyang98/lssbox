import numpy as np
import pytest
from nbodykit.source.catalog import UniformCatalog
from lssbox.pk import FFTPowerAP
from lssbox.recon import DisplacementSolver
from lssbox.recon import SafeFFTRecon
from .typing import atolT
from .typing import rtolT
from .typing import catalogT
from .typing import compare_ndarraysT
from .util import create_dict
from .util import load_csv_catalog


@pytest.mark.slow
def test_post_activate_vs_rebinning(
    catalog: catalogT, compare_ndarrays: compare_ndarraysT,
    atol: atolT, rtol: rtolT,
) -> None:
    data = load_csv_catalog(catalog)
    boxsize = data.attrs.pop('BoxSize')
    alpara, alperp = 0.8, 0.9
    rescale = np.array([alperp, alperp, alpara])
    nbar = data.csize / boxsize ** 3
    alpha = 10
    ran = UniformCatalog(nbar * alpha, BoxSize=boxsize, seed=42)
    power_kwargs = dict(
        mode='2d', Nmesh=512,
        los=[0, 0, 1], Nmu=100, poles=[0, 2, 4], dk=0.01,
        kmin=0, kmax=1.0, interlaced=True, compensated=True,
    )

    # solve displacement field
    solver = DisplacementSolver(
        dataA=data, ranA=ran, Nmesh=512, biasA=2.4, f=0.53,
        los=[0, 0, 1], R=10, revert_rsd_random=True, BoxSize=boxsize,
        resampler='cic', interlaced=True, compensated=True,
    )
    solver.run()

    # rebinning
    recon = SafeFFTRecon(
        data=data, ran=ran, Nmesh=512,
        s_d=solver.dis['dataA'], s_r=solver.dis['ranA'],
        bias=2.4, f=0.53, los=[0, 0, 1], R=10, revert_rsd_random=True,
        resampler='cic', interlaced=True, compensated=True, BoxSize=boxsize
    )
    ref = FFTPowerAP(
        recon, alpara=alpara, alperp=alperp, BoxSize=boxsize, **power_kwargs)
    ref_dict = create_dict(ref)
    ref_dict['shot'] *= (1 + 1 / alpha)

    # activate transform
    data['Position'] /= rescale  # type: ignore
    ran['Position'] /= rescale  # type: ignore
    recon = SafeFFTRecon(
        data=data, ran=ran, Nmesh=512,
        s_d=solver.dis['dataA'] / rescale, s_r=solver.dis['ranA'] / rescale,
        bias=2.4, f=0.53, los=[0, 0, 1], R=10, revert_rsd_random=True,
        resampler='cic', interlaced=True, compensated=True,
        BoxSize=list(boxsize / rescale)
    )
    get = FFTPowerAP(recon, BoxSize=list(boxsize / rescale), **power_kwargs)
    get_dict = create_dict(get)
    get_dict['shot'] *= (1 + 1 / alpha)

    compare_ndarrays(
        ref_dict, get_dict, default_tolerance={'atol': atol, 'rtol': 10 * rtol})
