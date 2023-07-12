"""
This trivial spring example targets to have the highest upward velocity. It is however only able to load a spring by
pulling downward and afterward to let it go so it gains velocity. It is designed to show how one can use the external
forces to interact with the body.
"""

import platform

from casadi import MX, vertcat
import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    ConfigureProblem,
    Objective,
    DynamicsFunctions,
    ObjectiveFcn,
    BoundsList,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
)


def custom_dynamic(
    states: MX, controls: MX, parameters: MX, stochastic_variables: MX, nlp: NonLinearProgram
) -> DynamicsEvaluation:
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
    stochastic_variables: MX
        The current stochastic variables of the system
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

    qddot = nlp.model.forward_dynamics(q, qdot, tau, force_vector)

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


def prepare_ocp(
    biorbd_model_path: str = "models/mass_point.bioMod",
    assume_phase_dynamics: bool = True,
):
    # BioModel path
    m = BiorbdModel(biorbd_model_path)
    m.set_gravity(np.array((0, 0, 0)))

    # Add objective functions (high upward velocity at end point)
    objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", index=0, weight=-1)

    # Dynamics
    dynamics = Dynamics(custom_configure, dynamic_function=custom_dynamic)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = [-1] * m.nb_q, [1] * m.nb_q
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = [-100] * m.nb_qdot, [100] * m.nb_qdot
    x_bounds["qdot"][:, 0] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * m.nb_tau, [0] * m.nb_tau

    return OptimalControlProgram(
        m,
        dynamics,
        n_shooting=30,
        phase_time=0.5,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
