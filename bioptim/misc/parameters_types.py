from typing import Any
import numpy as np


AnyIterable = list[Any] | tuple[Any]
IntIterable = list[int] | tuple[int]
StrIterable = list[str] | tuple[str]

StrIterableOptional = list[str] | tuple[str] | None

AnyDict = dict[str, Any]
IntDict = dict[str, int]

AnyDictOptional = dict[str, Any] | None

AnyList = list[Any]
IntList = list[int]
AnyListOrNpArray = list[Any] | np.ndarray

IntListOptional = list[int] | None

StrOrIterable = str | list[str]

IntOptional = int | None

StrOptional = str | None

AnyTuple = tuple[Any]
