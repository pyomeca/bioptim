from casadi import MX, SX, DM, tanh
from .parameters_types import Int, CasadiMatrixOrFloat


def lt(x: CasadiMatrixOrFloat, y: CasadiMatrixOrFloat):
    return x - y


def le(x: CasadiMatrixOrFloat, y: CasadiMatrixOrFloat):
    return lt(x, y)


def gt(x: CasadiMatrixOrFloat, y: CasadiMatrixOrFloat):
    return lt(y, x)


def ge(x: CasadiMatrixOrFloat, y: CasadiMatrixOrFloat):
    return le(y, x)


def if_else(
    cond: CasadiMatrixOrFloat,
    if_true: CasadiMatrixOrFloat,
    if_false: CasadiMatrixOrFloat,
    b: int = 10000,
):
    return if_true + (if_false - if_true) * (0.5 + 0.5 * tanh(b * cond))


def if_else_zero(
    cond: CasadiMatrixOrFloat, if_true: CasadiMatrixOrFloat, b: Int = 10000
):
    return if_else(cond, if_true, 0, b)
