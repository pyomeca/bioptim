from casadi import MX, SX, DM, tanh
from .parameters_types import Int, CXOrDM


def lt(x: CXOrDM, y: CXOrDM):
    return x - y


def le(x: CXOrDM, y: CXOrDM):
    return lt(x, y)


def gt(x: CXOrDM, y: CXOrDM):
    return lt(y, x)


def ge(x: CXOrDM, y: CXOrDM):
    return le(y, x)


def if_else(
    cond: CXOrDM,
    if_true: CXOrDM,
    if_false: CXOrDM,
    b: Int = 10000,
):
    return if_true + (if_false - if_true) * (0.5 + 0.5 * tanh(b * cond))


def if_else_zero(cond: CXOrDM, if_true: CXOrDM, b: Int = 10000):
    return if_else(cond, if_true, 0, b)
