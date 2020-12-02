import biorbd
import casadi as cas

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeOption,
    Problem,
    ObjectiveOption,
    DynamicsFunctions,
    Objective,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    ShowResult,
)

# This example load a mass on an upward spring and must have the greatest upward velocity at end point
# while only being able to pull on the system (the upward velocity being created by the spring).
# This example illustrates how you can use external forces that depends on the state of the system


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


# Model path
m = biorbd.Model("mass_point.bioMod")
m.setGravity(biorbd.Vector3d(0, 0, 0))

# Add objective functions (high upward velocity at end point)
objective_functions = ObjectiveOption(Objective.Mayer.MINIMIZE_STATE, index=1, weight=-1)

# Dynamics
dynamics = DynamicsTypeOption(custom_configure, dynamic_function=custom_dynamic)

# Path constraint
x_bounds = BoundsOption(QAndQDotBounds(m))
x_bounds[:, 0] = [0] * m.nbQ() + [0] * m.nbQdot()
x_bounds.min[:, 1] = [-1] * m.nbQ() + [-100] * m.nbQdot()
x_bounds.max[:, 1] = [1] * m.nbQ() + [100] * m.nbQdot()
x_bounds.min[:, 2] = [-1] * m.nbQ() + [-100] * m.nbQdot()
x_bounds.max[:, 2] = [1] * m.nbQ() + [100] * m.nbQdot()

# Initial guess
x_init = InitialGuessOption([0] * (m.nbQ() + m.nbQdot()))

# Define control path constraint
u_bounds = BoundsOption([[-100] * m.nbGeneralizedTorque(), [0] * m.nbGeneralizedTorque()])

u_init = InitialGuessOption([0] * m.nbGeneralizedTorque())
ocp = OptimalControlProgram(
        m,
        dynamics,
        number_shooting_points=30,
        phase_time=0.5,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )

# --- Solve the program --- #
sol = ocp.solve(show_online_optim=True)

# --- Show results --- #
result = ShowResult(ocp, sol)
result.animate()
