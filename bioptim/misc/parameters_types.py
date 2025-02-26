from typing import Any
import numpy as np
from typing import TypeAlias


Int: TypeAlias = int
Str: TypeAlias = str
Float: TypeAlias = float
Bool: TypeAlias = bool
Bytes: TypeAlias = bytes


AnyIterable: TypeAlias = list[Any] | tuple[Any, ...]
IntIterable: TypeAlias = list[int] | tuple[int]
StrIterable: TypeAlias = list[str] | tuple[str, ...]

StrIterableOptional: TypeAlias = list[str] | tuple[str, ...] | None

AnyDict: TypeAlias = dict[str, Any]
IntDict: TypeAlias = dict[str, int]

AnyDictOptional: TypeAlias = dict[str, Any] | None

AnyList: TypeAlias = list[Any]
IntList: TypeAlias = list[int]
AnyListOrNpArray: TypeAlias = list[Any] | np.ndarray
FloatIterableOrNpArray: TypeAlias = list[float] | tuple[float, ...] | np.ndarray

IntListOptional: TypeAlias = list[int] | None

StrOrIterable: TypeAlias = str | list[str]

IntOptional: TypeAlias = int | None

StrOptional: TypeAlias = str | None

BoolOptional: TypeAlias = bool | None

AnyTuple: TypeAlias = tuple[Any, ...]
