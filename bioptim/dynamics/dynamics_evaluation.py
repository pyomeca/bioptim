from ..misc.parameters_types import (
    CXOptional,
)


class DynamicsEvaluation:
    """
    Attributes
    ----------
    dxdt: MX | SX
        The derivative of the states xdot = f(t,x,u,p,s)
    defects: MX | SX
        defects of the dynamics for implicit transcription  f(t,x,u,p,s,xdot) = 0
    """

    def __init__(self, dxdt: CXOptional = None, defects: CXOptional = None):
        self.dxdt = dxdt
        self.defects = defects
