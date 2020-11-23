"""
File that shows an example of a custom constraint.
As an example, this custom constraint reproduces exactly the behavior of the ALIGN_MARKERS constraint.
"""
import biorbd
from casadi import vertcat

from bioptim import (
    Instant,
    OptimalControlProgram,
    DynamicsTypeOption,
    DynamicsType,
    ObjectiveOption,
    Objective,
    ConstraintList,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    ShowResult,
    OdeSolver,
)


def custom_func_align_markers(ocp, nlp, t, x, u, p, first_marker_idx, second_marker_idx):
    nq = nlp.shape["q"]
    val = []
    for v in x:
        q = v[:nq]
        markers = nlp.model.markers(q)
        first_marker = markers[:, first_marker_idx]
        second_marker = markers[:, second_marker_idx]
        val = vertcat(val, first_marker - second_marker)
    return val


def prepare_ocp(biorbd_model_path, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(custom_func_align_markers, instant=Instant.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(custom_func_align_markers, instant=Instant.END, first_marker_idx=0, second_marker_idx=2)

    # Path constraint
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    # Initial guess
    x_init = InitialGuessOption([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsOption(
        [[tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque()]
    )

    u_init = InitialGuessOption([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    model_path = "cube.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
