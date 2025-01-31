from bioptim import PenaltyController
from casadi import vertcat


def constraint_holonomic(
    controllers: PenaltyController,
):
    """
    Minimize the distance between two markers
    By default this function is quadratic, meaning that it minimizes distance between them.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    """

    q_u = controllers.states["q_u"]
    q_u_complete = q_u.mapping.to_second.map(q_u.cx)

    q_v = controllers.algebraic_states["q_v"]
    q_v_complete = q_v.mapping.to_second.map(q_v.cx)

    q = controllers.model.state_from_partition(q_u_complete, q_v_complete)

    holonomic_constraints = controllers.model.holonomic_constraints(q)

    for q_u_cx, q_v_cx in zip(q_u.cx_intermediates_list, q_v.cx_intermediates_list):
        q_u_complete = q_u.mapping.to_second.map(q_u_cx)
        q_v_complete = q_v.mapping.to_second.map(q_v_cx)
        q = controllers.model.state_from_partition(q_u_complete, q_v_complete)
        holonomic_constraints = vertcat(holonomic_constraints, controllers.model.holonomic_constraints(q))

    return holonomic_constraints
