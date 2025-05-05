from typing import Any
import numpy as np
from typing import TypeAlias
from casadi import MX, SX, DM
from .enums import Node

Int: TypeAlias = int
Range: TypeAlias = range
Str: TypeAlias = str
Float: TypeAlias = float
Bool: TypeAlias = bool
Tuple: TypeAlias = tuple
List: TypeAlias = list
Bytes: TypeAlias = bytes

IntorFloat: TypeAlias = int | float

AnyIterable: TypeAlias = list[Any] | tuple[Any, ...]
AnyIterableOptional: TypeAlias = list[Any] | tuple[Any, ...] | None
AnyIterableOrRange: TypeAlias = list | tuple | range
AnyIterableOrRangeOptional: TypeAlias = list | tuple | range | None
AnyIterableOrSlice: TypeAlias = slice | list | tuple
AnyIterableOrSliceOptional: TypeAlias = slice | list | tuple | None
AnySequence: TypeAlias = list | tuple | range | np.ndarray
AnySequenceOptional: TypeAlias = list | tuple | range | np.ndarray | None
IntIterable: TypeAlias = list[int] | tuple[int, ...]
IntIterableOptional: TypeAlias = list[int] | tuple[int, ...] | None
StrIterable: TypeAlias = list[str] | tuple[str, ...]

StrIterableOptional: TypeAlias = list[str] | tuple[str, ...] | None

AnyDict: TypeAlias = dict[str, Any]
AnyListorDict: TypeAlias = list[Any] | dict[str, Any]
IntDict: TypeAlias = dict[str, int]

AnyDictOptional: TypeAlias = dict[str, Any] | None
AnyListOptional: TypeAlias = list[Any] | None

AnyList: TypeAlias = list[Any]
BoolList: TypeAlias = list[bool]
IntList: TypeAlias = list[int]
FloatList: TypeAlias = list[float]
StrList: TypeAlias = list[str]
MXList: TypeAlias = list[MX]
DMList: TypeAlias = list[DM]
NpArrayList: TypeAlias = list[np.ndarray]
NpArray: TypeAlias = np.ndarray

NpArrayorFloat: TypeAlias = np.ndarray | float
NpArrayorFloatOptional: TypeAlias = np.ndarray | float | None
FloatIterableorNpArray: TypeAlias = list[float] | tuple[float, ...] | np.ndarray
FloatIterableorNpArrayorFloat: TypeAlias = FloatIterableorNpArray | float
IntIterableorNpArray: TypeAlias = list[int] | tuple[int, ...] | range | np.ndarray
IntIterableorNpArrayorInt: TypeAlias = int | IntIterableorNpArray

IntListOptional: TypeAlias = list[int] | None
FloatListOptional: TypeAlias = list[float] | None
StrListOptional: TypeAlias = list[str] | None
NpArrayListOptional: TypeAlias = list[np.ndarray] | None

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
DoubleIntTuple: TypeAlias = tuple[int, int]
DoubleFloatTuple: TypeAlias = tuple[float, float]
DoubleNpArrayTuple: TypeAlias = tuple[np.ndarray, np.ndarray]
StrTuple: TypeAlias = tuple[str, ...]

IntStrorIterable: TypeAlias = int | str | AnyIterable
IntorIterableOptional: TypeAlias = IntOptional | IntList | IntTuple
IntorNodeIterable: TypeAlias = tuple[Int | Node, ...] | list[Int | Node]

CX: TypeAlias = MX | SX
CXOptional: TypeAlias = MX | SX | None
CXorDM: TypeAlias = MX | SX | DM | float
CXorDMorFloatIterable: TypeAlias = FloatIterableorNpArray | MX | SX | DM
CXorDMorNpArray: TypeAlias = np.ndarray | MX | SX | DM
