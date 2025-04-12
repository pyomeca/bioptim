from typing import Any
import numpy as np
from typing import TypeAlias
from casadi import MX, SX, DM

Int: TypeAlias = int
Range: TypeAlias = range
Str: TypeAlias = str
Float: TypeAlias = float
Bool: TypeAlias = bool
Bytes: TypeAlias = bytes

AnyIterable: TypeAlias = list[Any] | tuple[Any, ...]
AnyIterableOrSlice: TypeAlias = slice | list | tuple
AnyIterableOrSliceOptional: TypeAlias = slice | list | tuple | None
IntIterable: TypeAlias = list[int] | tuple[int, ...]
StrIterable: TypeAlias = list[str] | tuple[str, ...]

StrIterableOptional: TypeAlias = list[str] | tuple[str, ...] | None

AnyDict: TypeAlias = dict[str, Any]
IntDict: TypeAlias = dict[str, int]

AnyDictOptional: TypeAlias = dict[str, Any] | None
AnyListOptional: TypeAlias = list[Any] | None

AnyList: TypeAlias = list[Any]
IntList: TypeAlias = list[int]
FloatList: TypeAlias = list[float]
StrList: TypeAlias = list[str]
MXList: TypeAlias = list[MX]
NpArray: TypeAlias = np.ndarray

NpArrayOrFloat: TypeAlias = np.ndarray | float
NpArrayOrFloatOptional: TypeAlias = np.ndarray | float | None
FloatIterableorNpArray: TypeAlias = list[float] | tuple[float, ...] | np.ndarray
IntIterableorNpArray: TypeAlias = list[int] | tuple[int, ...] | range | np.ndarray
IntIterableorNpArrayOrInt: TypeAlias = int | IntIterableorNpArray

IntListOptional: TypeAlias = list[int] | None
NpArrayListOptional: TypeAlias = list[np.ndarray] | None

StrOrIterable: TypeAlias = str | list[str]

IntOptional: TypeAlias = int | None

FloatOptional: TypeAlias = float | None

StrOptional: TypeAlias = str | None

BoolOptional: TypeAlias = bool | None

NpArrayOptional: TypeAlias = np.ndarray | None

AnyTuple: TypeAlias = tuple[Any, ...]
IntTuple: TypeAlias = tuple[int, ...]
DoubleIntTuple: TypeAlias = tuple[int, int]
StrTuple: TypeAlias = tuple[str, ...]

IntStrOrIterable: TypeAlias = int | str | AnyIterable

MXorSX: TypeAlias = MX | SX
MXorSXOptional: TypeAlias = MX | SX | None
CXOrDM: TypeAlias = MX | SX | DM | float
CXOrDMOrFloatIterable: TypeAlias = FloatIterableOrNpArray | MX | SX | DM
CXOrDMOrNpArray: TypeAlias = np.ndarray | MX | SX | DM
