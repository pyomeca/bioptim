import casadi


class ObjectiveFunction:
    @staticmethod
    def minimize_torque(nlp, weight=1):
        for i in range(nlp.ns):
            nlp.J += casadi.dot(nlp.U[i], nlp.U[i]) * nlp.dt * nlp.dt * weight

    @staticmethod
    def minimize_states(nlp, weight=1):
        raise RuntimeError("minimize_states objective function not implemented yet")

    @staticmethod
    def minimize_muscle(nlp, weight=1):
        raise RuntimeError("minimize_states objective function not implemented yet")

    @staticmethod
    def cyclic(nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")

