from typing import Any
import numpy as np
from typing import TypeAlias
from casadi import MX, SX, DM

Int: TypeAlias = int
Str: TypeAlias = str
Float: TypeAlias = float
Bool: TypeAlias = bool
Bytes: TypeAlias = bytes


AnyIterable: TypeAlias = list[Any] | tuple[Any, ...]
IntIterable: TypeAlias = list[int] | tuple[int, ...]
StrIterable: TypeAlias = list[str] | tuple[str, ...]

StrIterableOptional: TypeAlias = list[str] | tuple[str, ...] | None

AnyDict: TypeAlias = dict[str, Any]
IntDict: TypeAlias = dict[str, int]

AnyDictOptional: TypeAlias = dict[str, Any] | None

AnyList: TypeAlias = list[Any]
IntList: TypeAlias = list[int]
FloatIterableOrNpArray: TypeAlias = list[float] | tuple[float, ...] | np.ndarray
IntIterableOrNpArray: TypeAlias = list[int] | tuple[int, ...] | range | np.ndarray
IntIterableOrNpArrayOrInt: TypeAlias = int | IntIterableOrNpArray

IntListOptional: TypeAlias = list[int] | None

StrOrIterable: TypeAlias = str | list[str]

IntOptional: TypeAlias = int | None

StrOptional: TypeAlias = str | None

BoolOptional: TypeAlias = bool | None

AnyTuple: TypeAlias = tuple[Any, ...]
IntTuple: TypeAlias = tuple[int, ...]

IntStrOrIterable: TypeAlias = int | str | AnyIterable

CXOrDM: TypeAlias = MX | SX | DM | float
CXOrDMOrFloatIterable: TypeAlias = FloatIterableOrNpArray | MX | SX | DM
CXOrDMOrNpArray: TypeAlias = np.ndarray | MX | SX | DM
