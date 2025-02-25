from typing import Any

AnyDict = dict[str, Any]
AnyDictOptional = dict[str, Any] | None
IntDict = dict[str, int]

AnyList = list[Any]
IntList = list[int]
IntListOptional = list[int] | None

IntIterable = list[int] | tuple[int]

StrOrListOfStr = str | list[str]