import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    ObjectiveList,
    Objective,
    ConstraintList,
    Constraint,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    Instant,
    ShowResult,
    Data,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, time_min, time_max):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min, maximum=time_max)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].min[:, [0, -1]] = 0
    x_bounds[0].max[:, [0, -1]] = 0
    x_bounds[0].min[n_q - 1, -1] = 3.14
    x_bounds[0].max[n_q - 1, -1] = 3.14

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add([0] * (n_q + n_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * n_tau, [tau_max] * n_tau])
    u_bounds[0].min[n_tau - 1, :] = 0
    u_bounds[0].max[n_tau - 1, :] = 0

    u_init = InitialConditionsList()
    u_init.add([tau_init] * n_tau)

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
    )


if __name__ == "__main__":
    time_min = 0.6
    time_max = 1
    ocp = prepare_ocp(
        biorbd_model_path="pendulum.bioMod",
        final_time=2,
        number_shooting_points=50,
        time_min=time_min,
        time_max=time_max,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    print(f"The optimized phase time is: {param['time'][0, 0]}")

    result = ShowResult(ocp, sol)
    result.animate()
