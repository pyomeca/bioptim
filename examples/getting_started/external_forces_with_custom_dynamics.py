import biorbd
import casadi as cas

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeList,
    Problem,
    ObjectiveList,
    DynamicsFunctions,
    Objective,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    ShowResult,
    OdeSolver,
    BoundsList,
)


def custom_dynamic(states, controls, parameters, nlp):
    q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

    force_vector = cas.MX.zeros(6)
    force_vector[5] = 100 * q[0] ** 2

    f_ext = biorbd.VecBiorbdSpatialVector()
    f_ext.append(biorbd.SpatialVector(force_vector))
    qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()

    return qdot, qddot


def custom_configure(ocp, nlp):
    Problem.configure_q_qdot(nlp, as_states=True, as_controls=False)
    Problem.configure_tau(nlp, as_states=False, as_controls=True)
    Problem.configure_forward_dyn_func(ocp, nlp, custom_dynamic)


def prepare_ocp(biorbd_model_path, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    m = biorbd.Model(biorbd_model_path)
    m.setGravity(biorbd.Vector3d(0, 0, 0))

    # Problem parameters
    number_shooting_points = 30
    final_time = 0.5
    tau_min, tau_max, tau_init = -100, 0, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, index=1, weight=-1)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic)

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(QAndQDotBounds(m))
    X_bounds[0].min[:, 0] = [0] * m.nbQ() + [0] * m.nbQdot()
    X_bounds[0].max[:, 0] = [0] * m.nbQ() + [0] * m.nbQdot()
    X_bounds[0].min[:, 1] = [-1] * m.nbQ() + [-1000] * m.nbQdot()
    X_bounds[0].max[:, 1] = [1] * m.nbQ() + [1000] * m.nbQdot()
    X_bounds[0].min[:, 2] = [-1] * m.nbQ() + [-1000] * m.nbQdot()
    X_bounds[0].max[:, 2] = [1] * m.nbQ() + [1000] * m.nbQdot()

    # Initial guess
    x_init = InitialGuessOption([0] * (m.nbQ() + m.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsOption([[tau_min] * m.nbGeneralizedTorque(), [tau_max] * m.nbGeneralizedTorque()])

    u_init = InitialGuessOption([tau_init] * m.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        m,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        X_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    model_path = "MassPoint.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()

