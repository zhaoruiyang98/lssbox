"""
Common utilities for tests
"""
from __future__ import annotations
from nbodykit.base.catalog import CatalogSource
from nbodykit.source.catalog import CSVCatalog
from nbodykit.transform import StackColumns


def load_csv_catalog(path: str) -> CatalogSource:
    with open(path, 'r') as f:
        header = f.readline().split()
    if header[0] != '#':
        raise ValueError(f'missing # at the beginning of csv file: {path}')
    header = header[1:]
    usecols = ['xz', 'yz', 'zz']
    data = CSVCatalog(
        path, header, dtype='f8', usecols=usecols,
        delim_whitespace=True, skiprows=1,
        attrs={'BoxSize': 1000},
    )
    data['Position'] = StackColumns(data['xz'], data['yz'], data['zz'])
    return data


def create_dict(res):
    return dict(
        k=res.poles['k'],
        p0=res.poles['power_0'].real,
        p2=res.poles['power_2'].real,
        p4=res.poles['power_4'].real,
        shot=res.attrs['shotnoise'],
    )
