from typing import Any
from packaging.version import parse as parse_version


def check_version(tool_to_compare: Any, min_version: str, max_version: str, exclude_max: bool = True):
    """
    Check if the version of a certain module is okay. If the function exits, then everything is okay

    Parameters
    ----------
    tool_to_compare: Any
        The module to compare the version with
    min_version: str
        The minimum version accepted
    max_version: str
        The maximum version accepted
    exclude_max: bool
        If True, the max_version is excluded from the check
    """

    name = tool_to_compare.__name__
    ver = parse_version(tool_to_compare.__version__)
    if ver < parse_version(min_version):
        raise ImportError(f"{name} should be at least version {min_version}")
    elif ver >= parse_version(max_version) if exclude_max else ver > parse_version(max_version):
        raise ImportError(f"{name} should be lesser than version {max_version}")
