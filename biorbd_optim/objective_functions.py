import casadi


class ObjectiveFunction:
    @staticmethod
    def minimize_torque(nlp, weight=1):
        for i in range(nlp.ns):
            nlp.J += casadi.dot(nlp.U[i][nlp.model.nbMuscleTotal():nlp.model.nbGeneralizedTorque()], \
                                nlp.U[i][nlp.model.nbMuscleTotal():nlp.model.nbGeneralizedTorque()]) \
                                * nlp.dt * nlp.dt * weight

    @staticmethod
    def minimize_states(nlp, weight=1):
        for i in range(nlp.ns):
            nlp.J += casadi.dot(nlp.X[i], nlp.X[i]) * nlp.dt * nlp.dt * weight

    @staticmethod
    def minimize_muscle(nlp, weight=1):
        for i in range(nlp.ns):
            nlp.J += casadi.dot(nlp.U[i][:nlp.model.nbMuscleTotal()], nlp.U[i][:nlp.model.nbMuscleTotal()]) \
                                * nlp.dt * nlp.dt * weight

    @staticmethod
    def cyclic(nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")

