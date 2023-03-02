from casadi import MX, SX


class DynamicsEvaluation:
    """
    Attributes
    ----------
    dxdt: MX | SX
        The derivative of the states xdot = f(x,u,p)
    defects: MX | SX
        defects of the dynamics for implicit transcription  f(xdot,x,u,p) = 0
    """

    def __init__(self, dxdt: MX | SX = None, defects: MX | SX = None):
        self.dxdt = dxdt
        self.defects = defects
