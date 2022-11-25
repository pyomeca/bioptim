"""
This trivial spring example targets to have the highest upward velocity. It is however only able to load a spring by
pulling downward and afterward to let it go so it gains velocity. It is designed to show how one can use the external
forces to interact with the body.
"""

from casadi import MX, vertcat
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    ConfigureProblem,
    Objective,
    DynamicsFunctions,
    ObjectiveFcn,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
)


def custom_dynamic(states: MX, controls: MX, parameters: MX, nlp: NonLinearProgram) -> DynamicsEvaluation:
    """
    The dynamics of the system using an external force (see custom_dynamics for more explanation)

    Parameters
    ----------
    states: MX
        The current states of the system
    controls: MX
        The current controls of the system
    parameters: MX
        The current parameters of the system
    nlp: NonLinearProgram
        A reference to the phase of the ocp

    Returns
    -------
    The state derivative
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    force_vector = MX.zeros(6)
    force_vector[5] = 100 * q[0] ** 2

    f_ext = biorbd.VecBiorbdSpatialVector()
    f_ext.append(biorbd.SpatialVector(force_vector))
    qddot = nlp.model.forward_dynamics(q, qdot, tau, f_ext).to_mx()

    return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    The configuration of the dynamics (see custom_dynamics for more explanation)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase of the ocp
    """
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic)


def prepare_ocp(biorbd_model_path: str = "models/mass_point.bioMod"):
    # Model path
    m = BiorbdModel(biorbd_model_path)
    m.setGravity(np.array((0, 0, 0)))

    # Add objective functions (high upward velocity at end point)
    objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", index=0, weight=-1)

    # Dynamics
    dynamics = Dynamics(custom_configure, dynamic_function=custom_dynamic)

    # Path constraint
    x_bounds = QAndQDotBounds(m)
    x_bounds[:, 0] = [0] * m.nbQ() + [0] * m.nbQdot()
    x_bounds.min[:, 1] = [-1] * m.nbQ() + [-100] * m.nbQdot()
    x_bounds.max[:, 1] = [1] * m.nbQ() + [100] * m.nbQdot()
    x_bounds.min[:, 2] = [-1] * m.nbQ() + [-100] * m.nbQdot()
    x_bounds.max[:, 2] = [1] * m.nbQ() + [100] * m.nbQdot()

    # Initial guess
    x_init = InitialGuess([0] * (m.nbQ() + m.nbQdot()))

    # Define control path constraint
    u_bounds = Bounds([-100] * m.nb_generalized_torque(), [0] * m.nb_generalized_torque())

    u_init = InitialGuess([0] * m.nb_generalized_torque())
    return OptimalControlProgram(
        m,
        dynamics,
        n_shooting=30,
        phase_time=0.5,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


def main():
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
