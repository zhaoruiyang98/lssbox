from __future__ import annotations
from typing import Dict, Literal, Protocol
from typing_extensions import TypeAlias
from numpy.typing import ArrayLike

atolT: TypeAlias = float
rtolT: TypeAlias = float
catalogT: TypeAlias = str
ToleranceT: TypeAlias = Dict[Literal['atol', 'rtol'], float]


class compare_ndarraysT(Protocol):
    def __call__(
        self,
        ref_dct: dict[str, ArrayLike],
        data_dct: dict[str, ArrayLike],
        basename: str | None = None,
        fullpath: str | None = None,
        tolerances: dict[str, ToleranceT] | None = None,
        default_tolerance: ToleranceT | None = None,
    ) -> None:
        ...
