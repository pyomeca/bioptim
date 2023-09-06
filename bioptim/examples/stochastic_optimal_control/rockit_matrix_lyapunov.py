import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
import pickle

from bioptim import (
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    ObjectiveList,
    OptimalControlProgram,
    NonLinearProgram,
    DynamicsList,
    BoundsList,
    InterpolationType,
    SocpType,
    Node,
    ConstraintList,
    InitialGuessList,
    PenaltyController,
    ConfigureProblem,
    OdeSolver,
    StochasticBioModel,
)
from bioptim.examples.stochastic_optimal_control.rockit_model import RockitModel
from bioptim.examples.stochastic_optimal_control.common import get_m_init, get_cov_init

def configure_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, with_noise=False
        ),
    )

def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=2, n_collocation_points=nlp.model.polynomial_degree+1)
    ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=2)
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, with_noise=False
        ),
    )
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, with_noise=True
        ),
        allow_free_variables=True,
    )

def cost(controller: PenaltyController):
    q = controller.states["q"].cx_start
    return (q-3)**2  #ocp.add_objective(ocp.integral(sumsqr(x[0] - 3)))

def bound(t):
    return 2 + 0.1 * cas.cos(10 * t)

def path_constraint(controller: PenaltyController, dt, is_robustified: bool = False):
    t = controller.node_index * dt  # ocp.subject_to(x[0] <= bound(ocp.t) - sigma)
    q = controller.states["q"].cx_start
    sup = bound(t)
    if is_robustified:
        P = StochasticBioModel.reshape_to_matrix(controller.stochastic_variables["cov"].cx_start, controller.model.matrix_shape_cov)
        sigma = cas.sqrt(cas.horzcat(1, 0) @ P @ cas.vertcat(1, 0))
        sup -= sigma
    return q - sup


def prepare_ocp(
    final_time: float,
    n_shooting: int,
    polynomial_degree: int,
) -> OptimalControlProgram:
    """
    Step # 1: Solving the deterministic version of the problem to get the nominal trajectory.
    """

    bio_model = RockitModel()

    nb_q = bio_model.nb_q
    nb_u = bio_model.nb_u

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        cost,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        quadratic=False)


    # Constraints
    constraints = ConstraintList() #ocp.subject_to(ocp.at_t0(x) == vertcat(0.5, 0))
    # constraints.add(ConstraintFcn.TRACK_STATE, key="q", min_bound=0.5, max_bound=0.5, node=Node.START)
    # constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", min_bound=0, max_bound=0, node=Node.START)
    # ocp.subject_to(x[0] <= bound(ocp.t) - sigma)
    constraints.add(path_constraint,
                    dt=final_time/n_shooting,
                    is_robustified=False,
                    min_bound=-cas.inf,
                    max_bound=0,
                    node=Node.ALL)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_optimal_control_problem,
        dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, with_noise: bio_model.dynamics(
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            with_noise=with_noise,
        ),
        expand=True,
    )

    x_bounds = BoundsList()
    x_bounds["q"] = [-0.25] * nb_q, [cas.inf] * nb_q
    x_bounds["q"][:, 0] = 0.5  # Start  at 0.5
    x_bounds["qdot"] = [-cas.inf] * nb_q, [cas.inf] * nb_q
    x_bounds["qdot"][:, 0] = 0  # Start  at 0.5



    u_bounds = BoundsList()
    u_bounds["u"] = [-40] * nb_u, [40] * nb_u

    # Initial guesses
    # x_init = InitialGuessList()
    # u_init = InitialGuessList()



    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        # x_init=x_init,
        # u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre"),
        n_threads=1,
        assume_phase_dynamics=False,
    )


def prepare_socp(
    final_time: float,
    n_shooting: int,
    motor_noise_magnitude: np.ndarray,
    polynomial_degree: int,
    q_init: np.ndarray | None,
    qdot_init: np.ndarray,
    u_init: np.ndarray,
    m_init: np.ndarray | None = None,
    cov_init: np.ndarray | None = None,
    is_robustified: bool = False,
) -> StochasticOptimalControlProgram:

    """
    Step # 2-3: Solving the stochastic version of the problem to get the stochastic trajectory.
    """
    problem_type = SocpType.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre")

    bio_model = RockitModel(motor_noise_magnitude=motor_noise_magnitude, polynomial_degree=polynomial_degree)

    nb_q = bio_model.nb_q
    nb_qdot = bio_model.nb_qdot
    nb_x = nb_q + nb_qdot
    nb_u = bio_model.nb_u

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        cost,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        quadratic=False)


    # Constraints
    constraints = ConstraintList()
    constraints.add(path_constraint, # ocp.subject_to(x[0] <= bound(ocp.t) - sigma)
                    dt=final_time/n_shooting,
                    is_robustified=is_robustified,
                    min_bound=-cas.inf,
                    max_bound=0,
                    node=Node.ALL)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_stochastic_optimal_control_problem,
        dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, with_noise: bio_model.dynamics(
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            with_noise=with_noise,
        ),
        expand=True,
    )

    x_bounds = BoundsList()
    x_bounds["q"] = [-0.25] * nb_q, [cas.inf] * nb_q
    x_bounds["q"][:, 0] = 0.5  # Start  at 0.5
    x_bounds["qdot"] = [-cas.inf] * nb_q, [cas.inf] * nb_q
    x_bounds["qdot"][:, 0] = 0  # Start  at 0.5

    u_bounds = BoundsList()
    u_bounds["u"] = [-40] * nb_u, [40] * nb_u


    s_init = InitialGuessList()

    s_bounds = BoundsList()
    n_m = nb_x **2 * (polynomial_degree + 1)
    n_cov = nb_x **2
    n_stochastic = n_m + n_cov
    # Initial guesses
    # x_init = InitialGuessList()
    # u_init = InitialGuessList()

    if m_init is None:
        # m_init = np.ones((n_m, n_shooting+1)) * 0.01
        m_init = get_m_init(bio_model, n_stochastic, n_shooting, final_time, polynomial_degree, q_init, qdot_init, u_init)
    s_init.add(
        "m",
        initial_guess=m_init,
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "m",
        min_bound=[-cas.inf] * n_m,
        max_bound=[cas.inf] * n_m,
        interpolation=InterpolationType.CONSTANT,
    )

    P0 = np.diag([0.01 ** 2, 0.1 ** 2])

    if cov_init is None:
        cov_init_matrix = P0
        cov_init = get_cov_init(bio_model,
                                n_shooting,
                                n_stochastic,
                                polynomial_degree,
                                final_time,
                                q_init,
                                qdot_init,
                                u_init,
                                m_init,
                                cov_init_matrix,
                                motor_noise_magnitude)

    s_init.add(
        "cov",
        initial_guess=cov_init,
        interpolation=InterpolationType.EACH_FRAME,
    )

    s_bounds.add(
        "cov",
        min_bound=[-cas.inf] * n_cov,
        max_bound=[cas.inf] * n_cov,
        interpolation=InterpolationType.CONSTANT,
    )
    s_bounds["cov"][:, 0] = P0.flatten()


    # ocp.subject_to(ocp.at_t0(P) == P0)
    # ocp.set_initial(P, P0)

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        #x_init=x_init,
        #u_init=control_init,
        s_init=s_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        s_bounds=s_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=problem_type,
    )

def main():
    """
    Prepare, solve and plot the solution

    The problem is solved in 3 steps with a warm-start between each step:
    step #1: solve the deterministic version
    step #2: solve the stochastic version without the robustified constraint
    step #3: solve the stochastic version with the robustified constraint
    """

    # --- Prepare the ocp --- #
    bio_model = RockitModel()
    n_shooting = 40
    final_time = 1
    polynomial_degree = 5
    dt = final_time/n_shooting
    ts = np.arange(n_shooting+1)*dt
    motor_noise_magnitude = np.zeros(1)

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(100)

    ocp = prepare_ocp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
    )

    sol_ocp = ocp.solve(solver)
    tc = sol_ocp.time
    q_deterministic = sol_ocp.states["q"]
    qdot_deterministic = sol_ocp.states["qdot"]
    u_deterministic = sol_ocp.controls["u"]

    #sol_ocp.graphs()
    plt.figure()
    plt.plot(tc, bound(tc), label="bound")
    plt.plot(np.squeeze(tc), np.squeeze(q_deterministic), label="q_determinist")
    plt.step(ts, np.squeeze(u_deterministic/40), label="u_deterministic/40")


    socp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
        motor_noise_magnitude=motor_noise_magnitude,
        q_init=q_deterministic,
        qdot_init=qdot_deterministic,
        u_init=u_deterministic,
        is_robustified=False,
    )


    sol_socp = socp.solve(solver)
    q_stochastic = sol_socp.states["q"]
    qdot_stochastic = sol_socp.states["qdot"]
    u_stochastic = sol_socp.controls["u"]
    m_stochastic = sol_socp.stochastic_variables["m"]
    cov_stochastic = sol_socp.stochastic_variables["cov"]

    #sol_socp.graphs()

    plt.plot(np.squeeze(tc), np.squeeze(q_stochastic), '--', label="q_stochastic")
    plt.step(ts, np.squeeze(u_stochastic/40), '--', label="u_stochastic/40")
    plt.legend()

    o = np.array([[1, 0]])
    sigma = np.zeros((1, n_shooting+1))
    for i in range(n_shooting+1):
        Pi = np.reshape(cov_stochastic[:, i], (2, 2))
        sigma[:, i] = np.sqrt(o @ Pi @ o.T)
    plt.plot([ts, ts], np.squeeze([q_stochastic[:, ::polynomial_degree+1] - sigma, q_stochastic[:, ::polynomial_degree+1] + sigma]), 'k')

    plt.show()


    rsocp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
        motor_noise_magnitude=motor_noise_magnitude,
        q_init=q_stochastic,
        qdot_init=qdot_stochastic,
        u_init=u_stochastic,
        m_init=m_stochastic,
        cov_init=cov_stochastic,
        is_robustified=True,
    )
    sol_rsocp = rsocp.solve(solver)
    q_robustified = sol_rsocp.states["q"]
    qdot_robustified = sol_rsocp.states["qdot"]
    u_robustified = sol_rsocp.controls["u"]
    m_robustified = sol_rsocp.stochastic_variables["m"]
    cov_robustified = sol_rsocp.stochastic_variables["cov"]
    # robustified_data = {"q_robustified": q_robustified,
    #                     "qdot_robustified": qdot_robustified,
    #                     "u_robustified": u_robustified,
    #                     "m_robustified": m_robustified,
    #                     "cov_robustified": cov_robustified}
    #
    # with open('robustified.pkl', 'wb') as f:
    #     pickle.dump(robustified_data, f)
    # sol_rsocp.graphs()
    #

    # for i in range(n_shooting+1):
    #     cov = cov_stochastic[:,i].reshape((4, 4))
    #     plot_cov_ellipse(cov[:2, :2], [q_stochastic[0][i], q_stochastic[1][i]], alpha=0.25, color='blue')
    #
    plt.plot(q_robustified[0], q_robustified[1], "-b", label="Stochastic robustified")
    #
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    # plt.legend()
    # plt.savefig("output.png")
    # plt.show()


if __name__ == "__main__":
    main()
