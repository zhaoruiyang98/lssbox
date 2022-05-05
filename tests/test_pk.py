from nbodykit.algorithms import FFTPower
from lssbox.pk import FFTPowerAP
from .typing import atolT
from .typing import rtolT
from .typing import catalogT
from .typing import compare_ndarraysT
from .util import load_csv_catalog


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

    def create_dict(res):
        return dict(
            k=res.poles['k'],
            p0=res.poles['power_0'].real,
            p2=res.poles['power_2'].real,
            p4=res.poles['power_4'].real,
            shot=res.attrs['shotnoise'],
        )
    ref = FFTPower(data, **common_kwargs)
    ref_dict = create_dict(ref)
    get = FFTPowerAP(data, **common_kwargs)
    get_dict = create_dict(get)

    compare_ndarrays(
        ref_dict, get_dict, default_tolerance={"atol": atol, "rtol": rtol})
