from typing import Any


AnyIterable = list[Any] | tuple[Any]
AnyDict = dict[str, Any]
AnyDictOptional = dict[str, Any] | None
IntDict = dict[str, int]

AnyList = list[Any]
IntList = list[int]
IntListOptional = list[int] | None

IntIterable = list[int] | tuple[int]

StrOrIterable = str | list[str]

IntOptional = int | None
StrOptional = str | None