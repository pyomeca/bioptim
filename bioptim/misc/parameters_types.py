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
AnyIterableOptional: TypeAlias = list[Any] | tuple[Any, ...] | None
AnyIterableOrSlice: TypeAlias = slice | list | tuple
AnyIterableOrSliceOptional: TypeAlias = slice | list | tuple | None
AnySequence: TypeAlias = list | tuple | range | np.ndarray
AnySequenceOptional: TypeAlias = list | tuple | range | np.ndarray | None
IntIterable: TypeAlias = list[int] | tuple[int, ...]
StrIterable: TypeAlias = list[str] | tuple[str, ...]

StrIterableOptional: TypeAlias = list[str] | tuple[str, ...] | None

AnyDict: TypeAlias = dict[str, Any]
IntDict: TypeAlias = dict[str, int]

AnyDictOptional: TypeAlias = dict[str, Any] | None
AnyListOptional: TypeAlias = list[Any] | None

AnyList: TypeAlias = list[Any]
BoolList: TypeAlias = list[bool]
IntList: TypeAlias = list[int]
FloatList: TypeAlias = list[float]
StrList: TypeAlias = list[str]
MXList: TypeAlias = list[MX]
NpArrayList: TypeAlias = list[np.ndarray]
NpArray: TypeAlias = np.ndarray
NpArrayOrFloat: TypeAlias = np.ndarray | float
NpArrayOrFloatOptional: TypeAlias = np.ndarray | float | None
FloatIterableOrNpArray: TypeAlias = list[float] | tuple[float, ...] | np.ndarray
IntIterableOrNpArray: TypeAlias = list[int] | tuple[int, ...] | range | np.ndarray
IntIterableOrNpArrayOrInt: TypeAlias = int | IntIterableOrNpArray

IntListOptional: TypeAlias = list[int] | None

StrOrIterable: TypeAlias = str | list[str]

IntOptional: TypeAlias = int | None
IntOrStr: TypeAlias = int | str
IntOrStrOptional: TypeAlias = int | str | None

FloatOptional: TypeAlias = float | None

StrOptional: TypeAlias = str | None

BoolOptional: TypeAlias = bool | None

NpArrayOptional: TypeAlias = np.ndarray | None

AnyTuple: TypeAlias = tuple[Any, ...]
IntTuple: TypeAlias = tuple[int, ...]
StrTuple: TypeAlias = tuple[str, ...]

IntStrOrIterable: TypeAlias = int | str | AnyIterable

MXorSX: TypeAlias = MX | SX
MXorSXOptional: TypeAlias = MX | SX | None
CXOrDM: TypeAlias = MX | SX | DM | float
CXOrDMOrFloatIterable: TypeAlias = FloatIterableOrNpArray | MX | SX | DM
CXOrDMOrNpArray: TypeAlias = np.ndarray | MX | SX | DM
