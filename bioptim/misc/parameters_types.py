from typing import Any
import numpy as np


AnyIterable = list[Any] | tuple[Any]
AnyDict = dict[str, Any]
AnyDictOptional = dict[str, Any] | None
IntDict = dict[str, int]

AnyList = list[Any]
AnyListOrNpArray = list[Any] | np.ndarray
IntList = list[int]
IntListOptional = list[int] | None

IntIterable = list[int] | tuple[int]

StrIterable = list[str] | tuple[str]
StrIterableOptional = list[str] | tuple[str] | None
StrOrIterable = str | list[str]

IntOptional = int | None
StrOptional = str | None

AnyTuple = tuple[Any]
