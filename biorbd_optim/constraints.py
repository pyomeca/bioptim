import enum

from casadi import vertcat


class Constraint:
    @staticmethod
    class Type(enum.Enum):
        """
        Different conditions between markers and segments
        """
        MARKERS_TO_PAIR = 0

    @staticmethod
    class Instant(enum.Enum):
        START = 0
        MID = 1
        INTERMEDIATES = 2
        END = 3
        ALL = 4

    @staticmethod
    def add_constraints(nlp):
        for elem in nlp.constraints:

            if elem[1] == Constraint.Instant.START:
                x = [nlp.X[0]]
            elif elem[1] == Constraint.Instant.MID:
                if nlp.ns % 2 == 0:
                    raise(ValueError("Number of shooting points must be odd to use MID"))
                x = [nlp.X[nlp.ns//2+1]]
            elif elem[1] == Constraint.Instant.INTERMEDIATES:
                x = nlp.X[1:nlp.ns-1]
            elif elem[1] == Constraint.Instant.END:
                x = [nlp.X[nlp.ns]]
            elif elem[1] == Constraint.Instant.ALL:
                x = nlp.X
            else:
                continue

            if elem[0] == Constraint.Type.MARKERS_TO_PAIR:
                Constraint.__markers_to_pair(nlp, x, elem[2])

    @staticmethod
    def __markers_to_pair(nlp, X, idx_marker):
        for x in X:
            marker1 = nlp.model.marker(x[:nlp.model.nbQ()], idx_marker[0]).to_mx()
            marker2 = nlp.model.marker(x[:nlp.model.nbQ()], idx_marker[1]).to_mx()
            nlp.g = vertcat(nlp.g, marker1 - marker2)
            for i in range(3):
                nlp.g_bounds.min.append(0)
                nlp.g_bounds.max.append(0)

    @staticmethod
    def continuity_constraint(nlp):
        # Loop over shooting nodes
        for k in range(nlp.ns):
            # Create an evaluation node
            end_node = nlp.dynamics.call({"x0": nlp.X[k], "p": nlp.U[k]})["xf"]

            # Save continuity constraints
            nlp.g = vertcat(nlp.g, end_node - nlp.X[k+1])
            for i in range(nlp.nx):
                nlp.g_bounds.min.append(0)
                nlp.g_bounds.max.append(0)

        if nlp.is_cyclic_constraint:
            # Save continuity constraints between final integration and first node
            nlp.g = vertcat(nlp.g, nlp.X[nlp.ns] - nlp.X[0])
            for i in range(nlp.nx):
                nlp.g_bounds.min.append(0)
                nlp.g_bounds.max.append(0)
