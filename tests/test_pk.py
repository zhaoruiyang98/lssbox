import numpy as np
import pytest
from nbodykit.algorithms import FFTPower
from lssbox.pk import FFTPowerAP
from .typing import atolT
from .typing import rtolT
from .typing import catalogT
from .typing import compare_ndarraysT
from .util import create_dict
from .util import load_csv_catalog


@pytest.mark.slow
def test_FFTPowerAP_vs_FFTPower(
    catalog: catalogT, compare_ndarrays: compare_ndarraysT,
    atol: atolT, rtol: rtolT,
) -> None:
    data = load_csv_catalog(catalog)
    common_kwargs = dict(
        mode='2d', Nmesh=512, BoxSize=data.attrs['BoxSize'],
        los=[0, 0, 1], Nmu=100, poles=[0, 2, 4], dk=0.01,
        kmin=0, kmax=1.0,
    )

    ref = FFTPower(data, **common_kwargs)
    ref_dict = create_dict(ref)
    get = FFTPowerAP(data, **common_kwargs)
    get_dict = create_dict(get)

    compare_ndarrays(
        ref_dict, get_dict, default_tolerance={"atol": atol, "rtol": rtol})


@pytest.mark.slow
def test_pre_activate_vs_rebinning(
    catalog: catalogT, compare_ndarrays: compare_ndarraysT,
    atol: atolT, rtol: rtolT,
) -> None:
    data = load_csv_catalog(catalog)
    boxsize = data.attrs['BoxSize']
    alpara, alperp = 0.8, 0.9
    rescale = np.array([alperp, alperp, alpara])
    common_kwargs = dict(
        mode='2d', Nmesh=512,
        los=[0, 0, 1], Nmu=100, poles=[0, 2, 4], dk=0.01,
        kmin=0, kmax=1.0, interlaced=True, compensated=True,
    )
    ref = FFTPowerAP(
        data, BoxSize=boxsize, alpara=alpara, alperp=alperp, **common_kwargs)
    ref_dict = create_dict(ref)

    data['Position'] /= rescale  # type: ignore
    data.attrs.pop('BoxSize')
    get = FFTPowerAP(data, BoxSize=list(boxsize / rescale), **common_kwargs)
    get_dict = create_dict(get)

    compare_ndarrays(
        ref_dict, get_dict, default_tolerance={"atol": atol, "rtol": rtol})


@pytest.mark.fcompare
def test_clip(
    catalog: catalogT, compare_ndarrays: compare_ndarraysT,
    atol: atolT, rtol: rtolT,
) -> None:
    import dask.array as da
    data = load_csv_catalog(catalog)
    boxsize = data.attrs['BoxSize']
    alpara, alperp = 0.97, 0.97
    rescale = np.array([alperp, alperp, alpara])
    data.attrs.pop('BoxSize')
    mask = (data['Position'] < boxsize) & (
        data['Position'] > 0)  # type: ignore
    mask = da.prod(mask, axis=-1).astype(bool)
    data = data[mask]
    data['Position'] /= rescale  # type: ignore

    common_kwargs = dict(
        mode='2d', Nmesh=512,
        los=[0, 0, 1], Nmu=100, poles=[0, 2, 4], dk=0.01,
        kmin=0, kmax=1.0, interlaced=True, compensated=True,
    )

    ref = FFTPowerAP(data, BoxSize=list(boxsize / rescale), **common_kwargs)
    ref_dict = create_dict(ref)

    mask = (data['Position'] < boxsize) & (
        data['Position'] > 0)  # type: ignore
    mask = da.prod(mask, axis=-1).astype(bool)
    data = data[mask]
    get = FFTPowerAP(data, BoxSize=boxsize, **common_kwargs)
    get_dict = create_dict(get)
    compare_ndarrays(
        ref_dict, get_dict, default_tolerance={"atol": atol, "rtol": rtol})


@pytest.mark.fcompare
def test_rsd(
    catalog: catalogT, compare_ndarrays: compare_ndarraysT,
    atol: atolT, rtol: rtolT,
) -> None:
    import dask.array as da
    data = load_csv_catalog(catalog)
    boxsize = data.attrs['BoxSize']

    common_kwargs = dict(
        mode='2d', Nmesh=512, BoxSize=boxsize,
        los=[0, 0, 1], Nmu=100, poles=[0, 2, 4], dk=0.01,
        kmin=0, kmax=1.0, interlaced=True, compensated=True,
    )

    ref = FFTPowerAP(data, **common_kwargs)
    ref_dict = create_dict(ref)

    mask = (data['Position'] < boxsize) & (
        data['Position'] > 0)  # type: ignore
    mask = da.prod(mask, axis=-1).astype(bool)
    data = data[mask]
    get = FFTPowerAP(data, **common_kwargs)
    get_dict = create_dict(get)

    compare_ndarrays(
        ref_dict, get_dict, default_tolerance={"atol": atol, "rtol": rtol})
