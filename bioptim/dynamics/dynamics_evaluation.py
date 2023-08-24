from casadi import MX, SX


class DynamicsEvaluation:
    """
    Attributes
    ----------
    dxdt: MX | SX
        The derivative of the states xdot = f(t,x,u,p,s)
    defects: MX | SX
        defects of the dynamics for implicit transcription  f(t,x,u,p,s,xdot) = 0
    """

    def __init__(self, dxdt: MX | SX = None, defects: MX | SX = None):
        self.dxdt = dxdt
        self.defects = defects
