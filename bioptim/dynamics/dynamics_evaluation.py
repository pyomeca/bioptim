from typing import Union
from casadi import MX, SX


class DynamicsEvaluation:
    """
    Attributes
    ----------
    dxdt: Union[MX, SX]
        The derivative of the states xdot = f(x,u,p)
    defects: Union[MX, SX]
        defects of the dynamics for implicit transcription  f(xdot,x,u,p) = 0
    """

    def __init__(self, dxdt: Union[MX, SX] = None, defects: Union[MX, SX] = None):
        self.dxdt = dxdt
        self.defects = defects
