"""
This example replicates the results from "An approximate stochastic optimal control framework to simulate nonlinear
neuro-musculoskeletal models in the presence of noise"(https://doi.org/10.1371/journal.pcbi.1009338).

The task is to unfold the arm to reach a target further from the trunk.
Noise is added on the motor execution (wM) and on the feedback (wEE and wEE_dot).
The expected joint angles (x_mean) are optimized like in  deterministic OCP, but the covariance matrix is minimized to
reduce uncertainty. This covariance matrix is computed from the expected states.
"""
import platform
from casadi import SX, MX, vertcat
import numpy as np

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    BiorbdModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    VariableScalingList,
)



def forward_dynamics_with_friction(
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    nlp: NonLinearProgram,
    my_additional_factor=1,
) -> DynamicsEvaluation:

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    muscle_activation = DynamicsFunctions.get(nlp.controls["muscle"], controls)

    M = nlp.model.Inertia(q[:, i])
    joint_friction = np.array([[0.05, 0.025], [0.025, 0.05]])

    # ddtheta(:, i) = M\(T(:, i) + T_EXT(:, i) - C - B * qdot(:, i));

    # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot) * my_additional_factor
    ddq = nlp.model.forward_dynamics(q, qdot, tau)

    # the user has to choose if want to return the explicit dynamics dx/dt = f(x,u,p)
    # as the first argument of DynamicsEvaluation or
    # the implicit dynamics f(x,u,p,xdot)=0 as the second argument
    # which may be useful for IRK or COLLOCATION integrators
    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_tau(ocp, nlp, False, True)
    ConfigureProblem.configure_muscles(ocp, nlp, False, True)

    ConfigureProblem.configure_c(ocp, nlp)
    ConfigureProblem.configure_a(ocp, nlp)
    ConfigureProblem.configure_cov(ocp, nlp)
    ConfigureProblem.configure_w_motor(ocp, nlp)
    ConfigureProblem.configure_w_position_feedback(ocp, nlp)
    ConfigureProblem.configure_w_velocity_feedback(ocp, nlp)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    n_threads: int = 1,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscle", wight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STOCHASTIC_VARIABLE, key="cov", weight=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=forward_dynamics_with_friction)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    # TODO: add the initial position
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_muscles = bio_model.nb_muscles
    n_tau = bio_model.nb_tau
    u_bounds = Bounds([-100] * n_tau + [0] * n_muscles, [100] * n_tau + [1] * n_muscles)

    u_init = InitialGuess([0] * (n_tau + n_muscles))

    # This should probably be done automatically
    n_stochastic = n_q + n_q**2 + n_q**2 + n_q + n_q + n_q
    s_bounds = Bounds([-100] * n_stochastic, [100] * n_stochastic)
    s_init = InitialGuess([0] * n_stochastic)  # TODO: to be changed probable really bad

    # Variable scaling
    x_scaling = VariableScalingList()
    x_scaling.add("q", scaling=[1, 1])
    x_scaling.add("qdot", scaling=[1, 1])

    u_scaling = VariableScalingList()
    u_scaling.add("tau", scaling=[1, 1])
    u_scaling.add("muscles", scaling=[1, 1])

    # TODO: we should probably change the name stochastic_variables -> helper_variables ?
    # TODO: add end effector position and velocity correction

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        s_init=s_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        s_bounds=s_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        n_threads=n_threads,
        assume_phase_dynamics=False,  # TODO: see if it can be done with assume_phase_dynamics=True
    )


def main():

    # import bioviz
    # b = bioviz.Viz("models/LeuvenArmModel.bioMod")
    # b.exec()

    # --- Prepare the ocp --- #
    # TODO change the model to their model
    dt = 0.1  # 0.01
    final_time = 0.8
    n_shooting = int(final_time/dt)

    # TODO: devrait-il y avoir ns ou ns+1 P ?
    ocp = prepare_ocp(biorbd_model_path="models/LeuvenArmModel.bioMod", final_time=final_time, n_shooting=n_shooting)

    # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)

    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # platform.system() == "Linux"
    # sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol.animate()


if __name__ == "__main__":
    main()
