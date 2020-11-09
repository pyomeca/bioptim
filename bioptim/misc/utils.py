from packaging.version import parse as parse_version


def check_version(tool_to_compare, min_version, max_version):
    name = tool_to_compare.__name__
    ver = parse_version(tool_to_compare.__version__)
    if ver < parse_version(min_version):
        raise ImportError(f"{name} should be at least version {min_version}")
    elif ver >= parse_version(max_version):
        raise ImportError(f"{name} should be lesser than version {max_version}")
