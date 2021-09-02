from typing import Union

from casadi import MX, SX, DM, tanh


def lt(x: Union[MX, SX, DM, float], y: Union[MX, SX, DM, float]):
    return x - y


def le(x: Union[MX, SX, DM, float], y: Union[MX, SX, DM, float]):
    return lt(x, y)


def gt(x: Union[MX, SX, DM, float], y: Union[MX, SX, DM, float]):
    return lt(y, x)


def ge(x: Union[MX, SX, DM, float], y: Union[MX, SX, DM, float]):
    return le(y, x)


def if_else(
    cond: Union[MX, SX, DM, float],
    if_true: Union[MX, SX, DM, float],
    if_false: Union[MX, SX, DM, float],
    b: int = 10000,
):
    return if_true + (if_false - if_true) * (0.5 + 0.5 * tanh(b * cond))


def if_else_zero(cond: Union[MX, SX, DM, float], if_true: Union[MX, SX, DM, float], b: int = 10000):
    return if_else(cond, if_true, 0, b)
