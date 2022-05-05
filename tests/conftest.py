from __future__ import annotations
import numpy as np
import re
import pytest
from pathlib import Path
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from nbodykit import setup_logging


def pytest_addoption(parser):
    parser.addoption(
        "--fcompare", action="store_true", default=False,
        help="show failed comparison",
    )
    parser.addoption(
        "--atol", type=float, default=0,
        help="absolute tolerance, by default 0",
    )
    parser.addoption(
        "--rtol", type=float, default=1e-6,
        help="relative tolerance, by default 1e-6",
    )
    catalog = str(
        Path(__file__).parent.parent / 'data'
        / 'molino.z0.0.s8_p.nbody225.hod2_zrsd.ascii')
    parser.addoption(
        "--catalog", type=str, default=catalog,
        help=f"catalog file, by default {catalog}",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--fcompare"):
        skip = pytest.mark.skip(reason="need --fcompare option to run")
        for item in items:
            if "fcompare" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope='session')
def atol(request):
    yield request.config.getoption("--atol")


@pytest.fixture(scope='session')
def rtol(request):
    yield request.config.getoption("--rtol")


@pytest.fixture(scope='session')
def catalog(request):
    yield request.config.getoption("--catalog")


class DisableForceRegen:
    def __init__(self, request) -> None:
        self.request = request
        self.raw = request.config.getoption("force_regen")

    def __enter__(self):
        self.request.config.option.force_regen = False

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.request.config.option.force_regen = self.raw


@pytest.fixture(scope='function')
def compare_ndarrays(ndarrays_regression: NDArraysRegressionFixture):
    source_data_dir: Path | None = None

    def kernel(
        ref_dct, data_dct,
        basename=None, fullpath=None, tolerances=None, default_tolerance=None
    ):
        # ~~~~~~credit: pytest_regressions, please check the LICENSE file~~~~~~~
        __tracebackhide__ = True
        if not isinstance(ref_dct, dict):
            raise TypeError(
                "Only dictionaries with NumPy arrays or array-like objects are "
                "supported on ndarray_regression fixture.\n"
                "Object with type '{}' was given.".format(str(type(ref_dct)))
            )
        for key, array in ref_dct.items():
            assert isinstance(
                key, str
            ), "The dictionary keys must be strings. " "Found key with type '%s'" % (
                str(type(key))
            )
            ref_dct[key] = np.asarray(array)

        for key, array in ref_dct.items():
            if array.dtype.kind not in ["b", "i", "u", "f", "c", "U"]:
                raise TypeError(
                    "Only numeric or unicode data is supported on ndarrays_regression "
                    f"fixture.\nArray '{key}' with type '{array.dtype}' was given."
                )

        assert not (
            basename and fullpath), "pass either basename or fullpath, but not both"
        with_test_class_names = ndarrays_regression._with_test_class_names
        request = ndarrays_regression.request
        with_test_class_names = (
            with_test_class_names
            or request.config.getoption("with_test_class_names")
        )
        new_basename = basename
        if basename is None:
            if (request.node.cls is not None) and (with_test_class_names):
                new_basename = \
                    re.sub(r"[\W]", "_", request.node.cls.__name__) + "_"
            else:
                new_basename = ""
            new_basename += re.sub(r"[\W]", "_", request.node.name)

        extension = ".npz"
        if fullpath:
            filename = source_filename = Path(fullpath)
        else:
            filename = ndarrays_regression.datadir / \
                (new_basename + extension)  # type: ignore
            source_filename = \
                ndarrays_regression.original_datadir / \
                (new_basename + extension)  # type: ignore
        np.savez_compressed(str(filename), **ref_dct)
        nonlocal source_data_dir
        source_data_dir = source_filename.parent
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        with DisableForceRegen(request):
            ndarrays_regression.check(
                data_dct,
                basename=basename, fullpath=fullpath,
                tolerances=tolerances, default_tolerance=default_tolerance
            )

    yield kernel

    if source_data_dir is not None:
        if source_data_dir.exists():
            if not any(source_data_dir.iterdir()):
                source_data_dir.rmdir()


setup_logging()
