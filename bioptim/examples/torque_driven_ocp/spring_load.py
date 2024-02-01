"""
This trivial spring example targets to have the highest upward velocity. It is however only able to load a spring by
pulling downward and afterward to let it go so it gains velocity. It is designed to show how one can use the external
forces to interact with the body.
"""

import platform

from casadi import MX, vertcat, sign
import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    ConfigureProblem,
    ObjectiveList,
    DynamicsFunctions,
    ObjectiveFcn,
    BoundsList,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    PhaseDynamics,
    SolutionMerge,
)
from matplotlib import pyplot as plt


# TODO: scenario 1 does not work

    # scenarios are based on a Mayer term (at Tf)
    # 0: maximize upward speed - expected kinematics: negative torque to get as low as possible and release
    # 1: maximize downward speed - expected kinematics: positive torque to get as high as possible and release
    # 2: minimize quadratic speed - expected kinematics: no torque no move
    # 3: maximize quadratic speed - as in 1
    # 4-7 same as 0-3 but for COM

scenarios = {
    0: {"label": "max qdot(T)", "quad": False, "sign": -1, "tau_min": -100, "tau_max": 0, "check_tau": -1, "check_qdot(T)": 1,},
    1: {"label": "min qdot(T)", "quad": False, "sign": 1, "tau_min": 0, "tau_max": 100, "check_tau": 1, "check_qdot(T)": -1, },
    2: {"label": "min qdot(T)**2", "quad": True, "sign": 1, "tau_min": -100, "tau_max": 100, "check_tau": 1, "check_qdot(T)": 1,},
    3: {"label": "max qdot(T)**2", "quad": True, "sign": -1, "tau_min": -100, "tau_max": 0, "check_tau": -1, "check_qdot(T)": 1,},
    4: {"label": "max COMdot(T)", "quad": False, "sign": -1, "tau_min": -100, "tau_max": 0, "check_tau": -1, "check_qdot(T)": 1,},
    5: {"label": "min COMdot(T)", "quad": False, "sign": 1, "tau_min": 0, "tau_max": 100, "check_tau": 1,"check_qdot(T)": -1, },
    6: {"label": "min COMdot(T)**2", "quad": True, "sign": 1, "tau_min": -100, "tau_max": 100, "check_tau": 1,"check_qdot(T)": 1, },
    7: {"label": "max COMdot(T)**2", "quad": True, "sign": -1, "tau_min": -100, "tau_max": 0, "check_tau": -1,"check_qdot(T)": 1, },
}
# MINIMIZE_COM_VELOCITY


def custom_dynamic(
    time: MX, states: MX, controls: MX, parameters: MX, algebraic_states: MX, nlp: NonLinearProgram
) -> DynamicsEvaluation:
    """
    The dynamics of the system using an external force (see custom_dynamics for more explanation)

    Parameters
    ----------
    time: MX
        The current time of the system
    states: MX
        The current states of the system
    controls: MX
        The current controls of the system
    parameters: MX
        The current parameters of the system
    algebraic_states: MX
        The current algebraic states of the system
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
    stiffness = 100
    force_vector[5] = -sign(q[0]) * stiffness * q[0]**2  # traction-compression spring

    qddot = nlp.model.forward_dynamics(q, qdot, tau, [["Point", force_vector]])

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
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    phase_time: float = 0.5,
    n_shooting: float = 30,
    scenario=1,
):
    # BioModel path
    m = BiorbdModel(biorbd_model_path)
    m.set_gravity(np.array((0, 0, 0)))

    weight = 1

    # Add objective functions (high upward velocity at end point)
    objective_functions = ObjectiveList()

    if "qdot" in scenarios[scenario]["label"]:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="qdot",
            index=0,
            weight=weight * scenarios[scenario]["sign"],
            quadratic=scenarios[scenario]["quad"],
        )
    elif "COMdot" in scenarios[scenario]["label"]:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY,
            index=2,
            weight=weight * scenarios[scenario]["sign"],
            quadratic=scenarios[scenario]["quad"],
        )

    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=1e-5,
        quadratic=True,
    )


    # Dynamics
    dynamics = Dynamics(
        custom_configure,
        dynamic_function=custom_dynamic,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = [-1] * m.nb_q, [1] * m.nb_q
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = [-100] * m.nb_qdot, [100] * m.nb_qdot
    x_bounds["qdot"][:, 0] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = scenarios[scenario]["tau_min"] * m.nb_tau, scenarios[scenario]["tau_max"] * m.nb_tau

    return OptimalControlProgram(
        m,
        dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


def main():

    phase_time = 0.5
    n_shooting = 30
    fig, axs = plt.subplots(1, 3)

    for scenario in range(8):#in [1]: #
        print(scenarios[scenario]["label"])
        ocp = prepare_ocp(phase_time=phase_time, n_shooting=n_shooting, scenario=scenario)

        ocp.print(to_console=True, to_graph=False)

        # --- Solve the program --- #
        sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
        q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        qdot = sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
        tau = sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
        time = np.linspace(0,phase_time, n_shooting+1)
        eps = 1e-6


        axs[0].plot(time, q.flatten(), label=scenarios[scenario]["label"])
        axs[0].set_title("q")

        axs[1].plot(time, qdot.flatten(), label=scenarios[scenario]["label"])
        axs[1].set_title("qdot")

        axs[2].step(time, np.hstack([tau.flatten(), np.nan]),  label=scenarios[scenario]["label"])
        axs[2].set_title("tau")


        test_tau = tau * scenarios[scenario]["check_tau"] >= 0
        test_qdot = qdot.flatten()[-1] * scenarios[scenario]["check_qdot(T)"]
        if not np.all(test_tau):
            raise ValueError("Tau are not as expected.")
        if not test_qdot >= 0:
            raise ValueError("qdot(T) is not as expected")

        # --- Show results --- #
        sol.print_cost()
        # sol.graphs()
        # sol.animate()


        axs[2].legend()
    plt.show()
if __name__ == "__main__":
    main()
